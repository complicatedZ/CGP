"""
Usage:
    python FBAsimulation/construction_reaction_based_graph.py \
        --input_file FBAsolution/M9_sufficient_423_1_proteome_weakbg.pkl \
        --output_path FBAsolution/flow_graph_423_1e-5/
"""

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
# Add the project root (~/test/CGP) to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helper_function import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct mass-flow graphs from saved FBA flux solutions."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the pickle file containing all_met_flux.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory in which graph files will be saved.",
    )
    parser.add_argument(
        "--model_file",
        default="universal_model.mat",
        help="Path to the universal COBRA model (default: universal_model.mat).",
    )
    parser.add_argument(
        "--model_list_file",
        default="AGORA2_models/ModelList.xlsx",
        help="Path to ModelList.xlsx (default: AGORA2_models/ModelList.xlsx).",
    )
    return parser.parse_args()


def main(input_file, output_path, model_file="../universal_model.mat",
         model_list_file="../AGORA2_models/ModelList.xlsx"):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Load model and FBA results.
    cobra.Configuration().solver = "glpk"
    model = cobra.io.load_matlab_model(model_file)

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    all_met_flux = data["all_met_flux"]
    # Retained for compatibility with the input data structure.
    all_growth_rate = data.get("all_growth_rate")
    all_medium = data.get("all_random_medium")

    modelname_df = pd.read_excel(model_list_file)
    modelname = np.asarray(modelname_df["ModelName"])

    # Build the stoichiometric matrix.
    S_df = cobra.util.create_stoichiometric_matrix(
        model, array_type="DataFrame"
    )
    reaction_list = list(S_df.columns)
    metabolite_list = list(S_df.index)
    S = S_df.to_numpy()

    # Metabolites excluded from mass-flow graph construction.
    currency_mets = [
        'ctp[c]', 'cdp[c]', 'pi[c]', 'h[c]', 'gtp[c]', 'gdp[c]', 'itp[c]', 'idp[c]',
        'nadph[c]', 'nadp[c]', 'fadh2[c]', 'fad[c]', 'fmnh2[c]', 'fmn[c]', 'mql8[c]',
        'mqn8[c]', '2dmmql8[c]', '2dmmq8[c]', 'accoa[c]', 'coa[c]', 'nad[c]', 'nadh[c]',
        'atp[c]', 'adp[c]', 'ppi[c]', 'pppi[c]', 'q8[c]', 'q8h2[c]', 'cbi[c]', 'amp[c]',
        'udp[c]', 'cmp[c]', 'o2[c]', 'h2o[c]', 'co2[c]', 'dmbzid[c]', 'PGPm1[c]',
        'apoACP[c]', 'ACP[c]', 'malACP[c]', 'dnarep[c]', 'proteinsynth[c]', 'rnatrans[c]',

        'ctp[p]', 'cdp[p]', 'pi[p]', 'h[p]', 'gtp[p]', 'gdp[p]', 'itp[p]', 'idp[p]',
        'nadph[p]', 'nadp[p]', 'fadh2[p]', 'fad[p]', 'fmnh2[p]', 'fmn[p]', 'mql8[p]',
        'mqn8[p]', '2dmmql8[p]', '2dmmq8[p]', 'accoa[p]', 'coa[p]', 'nad[p]', 'nadh[p]',
        'atp[p]', 'adp[p]', 'ppi[p]', 'pppi[p]', 'q8[p]', 'q8h2[p]', 'cbi[p]', 'amp[p]',
        'udp[p]', 'cmp[p]', 'o2[p]', 'h2o[p]', 'co2[p]', 'dmbzid[p]', 'PGPm1[p]',
        'apoACP[p]', 'ACP[p]', 'malACP[p]', 'dnarep[p]', 'proteinsynth[p]', 'rnatrans[p]',

        'ctp[e]', 'cdp[e]', 'pi[e]', 'h[e]', 'gtp[e]', 'gdp[e]', 'itp[e]', 'idp[e]',
        'nadph[e]', 'nadp[e]', 'fadh2[e]', 'fad[e]', 'fmnh2[e]', 'fmn[e]', 'mql8[e]',
        'mqn8[e]', '2dmmql8[e]', '2dmmq8[e]', 'accoa[e]', 'coa[e]', 'nad[e]', 'nadh[e]',
        'atp[e]', 'adp[e]', 'ppi[e]', 'pppi[e]', 'q8[e]', 'q8h2[e]', 'cbi[e]', 'amp[e]',
        'udp[e]', 'cmp[e]', 'o2[e]', 'h2o[e]', 'co2[e]', 'dmbzid[e]', 'PGPm1[e]',
        'apoACP[e]', 'ACP[e]', 'malACP[e]', 'dnarep[e]', 'proteinsynth[e]', 'rnatrans[e]',

        'h2o[p]', 'co2[p]', 'h[p]', 'pi[p]', 'h2o[e]', 'co2[e]', 'biomass[e]',
    ]

    metabolite_index = {met: idx for idx, met in enumerate(metabolite_list)}
    for met in currency_mets:
        idx = metabolite_index.get(met)
        if idx is not None:
            S[idx, :] = 0

    # Construct and save a graph for each flux entry.
    total_groups = len(all_met_flux)
    for i, flux_group in enumerate(all_met_flux):
        print(f"\nProcessing flux group {i + 1}/{total_groups}")

        for j, met_flux in enumerate(flux_group):
            try:
                if j >= len(modelname):
                    print(f"[WARNING] No model name for index {j}; skipping.")
                    continue

                model_label = str(modelname[j])[1:-5]
                output_stem = f"{i}_{model_label}"

                print(f"  Model {j + 1}/{len(flux_group)}: {model_label}")

                M1 = construct_mass_flow_graph(S, met_flux, reaction_list)
                sparse_M1 = sparse.csr_matrix(M1)
                sparse.save_npz(
                    os.path.join(output_path, output_stem + ".npz"),
                    sparse_M1,
                )

                G1 = construct_networkx_graph(M1, reaction_list)
                with open(
                    os.path.join(output_path, output_stem + ".pkl"), "wb"
                ) as f:
                    pickle.dump(G1, f, protocol=pickle.HIGHEST_PROTOCOL)

                del M1, sparse_M1, G1
                gc.collect()

            except Exception as exc:
                print(f"[ERROR] Flux group {i}, model index {j}: {exc}")
                continue

    print("\nFinished processing.")


if __name__ == "__main__":
    args = parse_args()
    main(
        input_file=args.input_file,
        output_path=args.output_path,
        model_file=args.model_file,
        model_list_file=args.model_list_file,
    )