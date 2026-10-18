#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import pfba


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_DIR = PROJECT_ROOT / "FBAsolution" / "Perturb_P"
OUTPUT_DIR = PROJECT_ROOT / "FBAsolution" / "Perturb_P"
OPT_EXCHANGES_FILE = PROJECT_ROOT / "opt_exchanges.csv"
UNIVERSAL_MODEL_FILE = (
    PROJECT_ROOT / "random_walk" / "universal_model_noegcs_nonreverse_fixed.xml"
)

SHARED_MEDIUM = {
    'EX_mn2(e)': 1000.0, 'EX_cu2(e)': 1000.0, 'EX_pi(e)': 1000.0,
    'EX_cobalt2(e)': 1000.0, 'EX_h(e)': 1000.0, 'EX_mg2(e)': 1000.0,
    'EX_co2(e)': 1000.0, 'EX_o2(e)': 1000.0, 'EX_cl(e)': 1000.0,
    'EX_zn2(e)': 1000.0, 'EX_so4(e)': 1000.0, 'EX_fe3(e)': 1000.0,
    'EX_h2o(e)': 1000.0, 'EX_k(e)': 1000.0, 'EX_nh4(e)': 1000.0,
    'EX_ca2(e)': 1000.0, 'EX_na1(e)': 1000.0, 'EX_fe2(e)': 1000.0
}


def get_input_path(run_index: int):
    return INPUT_DIR / f"M9_sufficient_423_1_proteome_{run_index}_by_model.pkl"


def get_output_path(run_index: int):
    return OUTPUT_DIR / f"M9_sufficient_423_1_proteome_{run_index}_unimodel.pkl"


def load_met_fluxes(run_index: int):
    input_path = get_input_path(run_index)
    with input_path.open("rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data["all_met_flux"][0]


def load_opt_exchanges():
    return set(np.array(pd.read_csv(OPT_EXCHANGES_FILE))[:, 1])


def add_proteome_constraint_once(model, flux_limit: float):
    biomass_reactions = {
        rec.id
        for rec in model.reactions
        if "bio" in rec.id and "EX_" not in rec.id and "pbio" not in rec.id
    }

    abs_flux_vars = []
    for reaction in model.reactions:
        if reaction.id in biomass_reactions or reaction.id in SHARED_MEDIUM:
            continue

        abs_flux_var_name = f"abs_flux_{reaction.id}"
        if abs_flux_var_name in model.variables:
            abs_flux_vars.append(model.variables[abs_flux_var_name])
            continue

        abs_flux_var = model.problem.Variable(abs_flux_var_name, lb=0)
        abs_flux_vars.append(abs_flux_var)
        model.add_cons_vars(
            [
                model.problem.Constraint(
                    abs_flux_var - reaction.flux_expression,
                    lb=0,
                ),
                model.problem.Constraint(
                    abs_flux_var + reaction.flux_expression,
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


def close_sink_reactions(model):
    for reaction in model.reactions:
        if "sink" in reaction.id:
            reaction.lower_bound = 0


def build_constrained_universal_model(flux_limit: float):
    model = cobra.io.read_sbml_model(str(UNIVERSAL_MODEL_FILE))
    add_proteome_constraint_once(model, flux_limit)
    close_sink_reactions(model)
    return model


def apply_medium_from_species_flux(model, met_flux, opt_exchanges):
    model.medium = {}

    for rec_id, flux in met_flux:
        if rec_id in opt_exchanges and flux < 0:
            model.reactions.get_by_id(rec_id).lower_bound = flux

    for rec_id in SHARED_MEDIUM:
        model.reactions.get_by_id(rec_id).lower_bound = -1000


def get_biomass_id(met_flux):
    biomass_ids = [
        rec_id
        for rec_id, _ in met_flux
        if "bio" in rec_id and "EX_" not in rec_id and "pbio" not in rec_id
    ]
    if not biomass_ids:
        return None
    return biomass_ids[0]


def simulate_one_species(model, met_flux, opt_exchanges, species_index: int):
    if met_flux == []:
        return 0, 0, 0, [], []

    biomass_id = get_biomass_id(met_flux)
    if biomass_id is None:
        print(f"No biomass reaction found for species index {species_index}")
        return 0, 0, 0, [], []

    model.objective = model.reactions.get_by_id(biomass_id)
    apply_medium_from_species_flux(model, met_flux, opt_exchanges)

    try:
        pfba_solution = pfba(model)
        growth_rate = pfba_solution.fluxes[biomass_id]
    except Exception as exc:
        print(f"pFBA failed for species index {species_index}: {exc}")
        return 0, 0, 0, [], []

    total_intake = 0
    total_nutrient_count = 0
    met_flux_2 = []

    if growth_rate > 0:
        for reaction in model.exchanges:
            flux_value = pfba_solution.fluxes[reaction.id]
            if flux_value < -1e-2 and reaction.id not in SHARED_MEDIUM:
                total_intake -= flux_value
                total_nutrient_count += 1
        met_flux_2 = [(rec.id, pfba_solution.fluxes[rec.id]) for rec in model.reactions]

    return total_intake, growth_rate, total_nutrient_count, model.medium, met_flux_2


def simulate_file(run_index: int, flux_limit: float):
    all_met_flux = load_met_fluxes(run_index)
    opt_exchanges = load_opt_exchanges()
    model = build_constrained_universal_model(flux_limit)

    species_count = len(all_met_flux)
    all_total_intake = np.zeros(species_count)
    all_growth_rate = np.zeros(species_count)
    all_total_nutrient_count = np.zeros(species_count)
    all_random_medium = []
    all2_met_flux = np.empty(species_count, dtype=object)

    for species_index, met_flux in enumerate(all_met_flux):
        print(f"Run {run_index}: species {species_index + 1}/{species_count}")
        (
            total_intake,
            growth_rate,
            total_nutrient_count,
            random_medium,
            met_flux_2,
        ) = simulate_one_species(model, met_flux, opt_exchanges, species_index)

        all_total_intake[species_index] = total_intake
        all_growth_rate[species_index] = growth_rate
        all_total_nutrient_count[species_index] = total_nutrient_count
        all_random_medium.append(random_medium)
        all2_met_flux[species_index] = met_flux_2

    return {
        "all_total_intake": all_total_intake,
        "all_growth_rate": all_growth_rate,
        "all_total_nutrient_count": all_total_nutrient_count,
        "all_random_medium": all_random_medium,
        "all_met_flux": all2_met_flux,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run universal model simulations for one by_model file. "
            "The positional i selects "
            "FBAsolution/Perturb_P/M9_sufficient_423_1_proteome_{i}_by_model.pkl."
        )
    )
    parser.add_argument("i", type=int, help="input file index")
    parser.add_argument(
        "--flux-limit",
        type=float,
        default=1e3,
        help="proteome constraint upper bound added once to the universal model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_to_save = simulate_file(args.i, args.flux_limit)

    output_path = get_output_path(args.i)
    with output_path.open("wb") as pickle_file:
        pickle.dump(data_to_save, pickle_file)

    print(f"Saved run {args.i} result to {output_path}")


if __name__ == "__main__":
    main()
