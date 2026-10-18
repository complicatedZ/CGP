#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import random
import re
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import pfba


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

SHARED_MEDIUM = {
    'EX_mn2(e)': 1000.0, 'EX_cu2(e)': 1000.0, 'EX_pi(e)': 1000.0,
    'EX_cobalt2(e)': 1000.0, 'EX_h(e)': 1000.0, 'EX_mg2(e)': 1000.0,
    'EX_co2(e)': 1000.0, 'EX_o2(e)': 1000.0, 'EX_cl(e)': 1000.0,
    'EX_zn2(e)': 1000.0, 'EX_so4(e)': 1000.0, 'EX_fe3(e)': 1000.0,
    'EX_h2o(e)': 1000.0, 'EX_k(e)': 1000.0, 'EX_nh4(e)': 1000.0,
    'EX_ca2(e)': 1000.0, 'EX_na1(e)': 1000.0, 'EX_fe2(e)': 1000.0
}


def find_objective_rec(objective_expression_str: str):
    matches = re.findall(r"bio[^\s\*]+", objective_expression_str)
    if matches:
        return matches[0]
    return None


def load_selected_model_from_directory(directory: Path, excel_file: Path):
    model_names_df = pd.read_excel(excel_file, sheet_name="Sheet1")

    models = []
    for model_name in model_names_df["ModelName"]:
        filename = f"{model_name}"
        model_path = directory / filename[1:-1]

        if model_path.exists():
            print(f"Found model: {filename}")
            models.append(model_path)
        else:
            print(f"File {filename} does not exist in the directory.")

    return models


def build_optional_exchanges(uni_model, drugs_file: Path):
    df = pd.read_excel(drugs_file, header=None)
    drug_list = df[0].tolist()

    return [
        rec.id
        for rec in uni_model.exchanges
        if rec.id not in SHARED_MEDIUM
        and rec.id != "EX_biomass(e)"
        and rec.id[3:-3] not in drug_list
    ]


def generate_random_medium(opt_exchanges, medium_count: int):
    if medium_count > len(opt_exchanges):
        raise ValueError(
            f"medium_count={medium_count} is larger than available optional "
            f"exchanges ({len(opt_exchanges)})."
        )

    sampled_exchanges = random.sample(opt_exchanges, medium_count)
    random_medium = SHARED_MEDIUM.copy()

    for exchange in opt_exchanges:
        random_medium[exchange] = 1e-2

    for exchange in sampled_exchanges:
        random_medium[exchange] = 20

    return random_medium


def build_random_media(opt_exchanges, medium_count: int, n: int):
    random_media = []
    for i in range(n):
        random.seed(1000 * i)
        random_media.append(generate_random_medium(opt_exchanges, medium_count))
    return random_media


def resolve_flux_limits(args, model_count: int):
    if args.flux_limits is not None:
        flux_limits = np.asarray(args.flux_limits, dtype=float)
        if flux_limits.size == 1:
            return np.repeat(flux_limits[0], model_count)
        if flux_limits.size != model_count:
            raise ValueError(
                f"Expected 1 or {model_count} flux limits, got {flux_limits.size}."
            )
        return flux_limits

    rng = np.random.default_rng(args.seed_base)
    return rng.uniform(args.lower_flux_limit, args.upper_flux_limit, size=model_count)


def add_total_abs_flux_constraint(model, flux_limit: float, w_dict):
    biomass_reaction = find_objective_rec(str(model.objective.expression))
    abs_flux_vars = []

    for reaction in model.reactions:
        if reaction.id == biomass_reaction:
            continue

        abs_flux_var_name = f"abs_flux_{reaction.id}"
        if abs_flux_var_name in model.variables:
            abs_flux_vars.append(model.variables[abs_flux_var_name])
            continue

        w = w_dict.get(reaction.id, 1)
        abs_flux_var = model.problem.Variable(abs_flux_var_name, lb=0)
        abs_flux_vars.append(abs_flux_var)
        model.add_cons_vars(
            [
                model.problem.Constraint(
                    abs_flux_var - w * reaction.flux_expression,
                    lb=0,
                ),
                model.problem.Constraint(
                    abs_flux_var + w * reaction.flux_expression,
                    lb=0,
                ),
            ]
        )

    if "total_abs_flux_constraint" not in model.constraints:
        total_abs_flux_constraint = model.problem.Constraint(
            sum(abs_flux_vars),
            ub=flux_limit,
            name="total_abs_flux_constraint",
        )
        model.add_cons_vars([total_abs_flux_constraint])

    return biomass_reaction


def apply_medium(model, random_medium):
    model.medium = {}
    for rec_id, flux in random_medium.items():
        try:
            model.reactions.get_by_id(rec_id).lower_bound = -flux
        except KeyError:
            continue


def run_one_medium(model, random_medium, biomass_reaction):
    apply_medium(model, random_medium)

    try:
        pfba_solution = pfba(model)
        growth_rate = pfba_solution.fluxes[biomass_reaction]
    except Exception as exc:
        print(f"pFBA failed: {exc}")
        return 0, 0, 0, []

    total_abs_flux = sum(abs(flux) for flux in pfba_solution.fluxes)
    print(total_abs_flux)

    total_intake = 0
    total_nutrient_count = 0
    for reaction in model.exchanges:
        flux_value = pfba_solution.fluxes[reaction.id]
        if flux_value < -1e-2 and reaction.id not in SHARED_MEDIUM:
            total_intake -= flux_value
            total_nutrient_count += 1

    met_flux = [(rec.id, pfba_solution.fluxes[rec.id]) for rec in model.reactions]
    return total_nutrient_count, growth_rate, total_intake, met_flux


def run_model_across_media(model_path: Path, random_media, flux_limit: float, w_dict):
    model = cobra.io.load_matlab_model(str(model_path))
    biomass_reaction = add_total_abs_flux_constraint(model, flux_limit, w_dict)

    results = []
    for i, random_medium in enumerate(random_media):
        print(f"    Medium {i + 1}/{len(random_media)}")
        results.append(run_one_medium(model, random_medium, biomass_reaction))
    
    del model  # Free memory

    return results


def run_simulation(args):
    if args.solver:
        cobra.Configuration().solver = args.solver

    uni_model = cobra.io.load_matlab_model(str(PROJECT_ROOT / "universal_model.mat"))
    opt_exchanges = build_optional_exchanges(uni_model, PROJECT_ROOT / "AGORA2_models" / "drugs.xlsx")
    models = load_selected_model_from_directory(
        directory=PROJECT_ROOT / "AGORA2_models",
        excel_file=PROJECT_ROOT / "AGORA2_models" / "ModelList.xlsx",
    )

    m = len(models)
    n = args.n
    medium_count = args.medium_count
    w_dict = {}

    all_random_medium = build_random_media(opt_exchanges, medium_count, n)
    flux_limits = resolve_flux_limits(args, m)

    all_total_intake = np.zeros((n, m))
    all_growth_rate = np.zeros((n, m))
    all_total_nutrient_count = np.zeros((n, m))
    all_met_flux = np.empty((n, m), dtype=object)
    all_flux_limits = np.tile(flux_limits, (n, 1))

    for j, model_path in enumerate(models):
        print(f"Model {j + 1}/{m}: {model_path.name}")
        print(f"  flux_limit={flux_limits[j]}")
        model_results = run_model_across_media(
            model_path,
            all_random_medium,
            flux_limits[j],
            w_dict,
        )

        for i, (total_nutrient_count, growth_rate, total_intake, met_flux) in enumerate(
            model_results
        ):
            all_total_nutrient_count[i, j] = total_nutrient_count
            all_growth_rate[i, j] = growth_rate
            all_total_intake[i, j] = total_intake
            all_met_flux[i, j] = met_flux

    data_to_save = {
        "all_total_intake": all_total_intake,
        "all_growth_rate": all_growth_rate,
        "all_total_nutrient_count": all_total_nutrient_count,
        "all_flux_limits": all_flux_limits,
        "all_random_medium": all_random_medium,
        "all_met_flux": all_met_flux,
    }

    output_path = (
        args.output_dir
        / f"M9_sufficient_{medium_count}_{n}_proteome_{args.seed_base}_by_model.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as pickle_file:
        pickle.dump(data_to_save, pickle_file)

    print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run across-species simulations, adding constraints once per model."
    )
    parser.add_argument("-n", "--n", type=int, default=1, help="random medium repeats")
    parser.add_argument(
        "-m",
        "--medium-count",
        type=int,
        default=423,
        help="number of optional exchanges sampled into the medium",
    )
    parser.add_argument(
        "--flux-limits",
        "--flux_limits",
        type=float,
        nargs="+",
        default=None,
        help="upper flux limit for each model; one value is broadcast to all models",
    )
    parser.add_argument(
        "--lower-flux-limit",
        type=float,
        default=1000,
        help="lower bound for randomly generated per-model flux limits",
    )
    parser.add_argument(
        "--upper-flux-limit",
        type=float,
        default=1000,
        help="upper bound for randomly generated per-model flux limits",
    )
    parser.add_argument(
        "--solver",
        "--solve",
        default="glpk",
        help="COBRA solver name; use an empty string to keep COBRA's default",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="seed for reproducible flux limits; media use the original 1000*i seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "FBAsolution",
        help="directory for the output pickle file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_simulation(parse_args())
