# ruff: noqa: B008

import pickle
from pathlib import Path
import pandas as pd
from mmore.index.implementations.graphrag import IndexerArtifacts



def save_artifacts(artifacts: IndexerArtifacts, path: Path):
    artifacts.entities.to_parquet(f"{path}/entities.parquet")
    artifacts.relationships.to_parquet(f"{path}/relationships.parquet")
    artifacts.text_units.to_parquet(f"{path}/text_units.parquet")
    artifacts.communities_reports.to_parquet(f"{path}/communities_reports.parquet")

    if artifacts.merged_graph is not None:
        with path.joinpath("merged-graph.pickle").open("wb") as fp:
            pickle.dump(artifacts.merged_graph, fp)

    if artifacts.summarized_graph is not None:
        with path.joinpath("summarized-graph.pickle").open("wb") as fp:
            pickle.dump(artifacts.summarized_graph, fp)

    if artifacts.communities is not None:
        with path.joinpath("community_info.pickle").open("wb") as fp:
            pickle.dump(artifacts.communities, fp)


def load_artifacts(path: Path) -> IndexerArtifacts:
    entities = pd.read_parquet(f"{path}/entities.parquet")
    relationships = pd.read_parquet(f"{path}/relationships.parquet")
    text_units = pd.read_parquet(f"{path}/text_units.parquet")
    communities_reports = pd.read_parquet(f"{path}/communities_reports.parquet")

    merged_graph = None
    summarized_graph = None
    communities = None

    merged_graph_pickled = path.joinpath("merged-graph.pickle")
    if merged_graph_pickled.exists():
        with merged_graph_pickled.open("rb") as fp:
            merged_graph = pickle.load(fp)  # noqa: S301

    summarized_graph_pickled = path.joinpath("summarized-graph.pickle")
    if summarized_graph_pickled.exists():
        with summarized_graph_pickled.open("rb") as fp:
            summarized_graph = pickle.load(fp)  # noqa: S301

    community_info_pickled = path.joinpath("community_info.pickle")
    if community_info_pickled.exists():
        with community_info_pickled.open("rb") as fp:
            communities = pickle.load(fp)  # noqa: S301

    return IndexerArtifacts(
        entities,
        relationships,
        text_units,
        communities_reports,
        merged_graph=merged_graph,
        summarized_graph=summarized_graph,
        communities=communities,
    )


def get_artifacts_dir_name(model: str) -> str:
    return f"artifacts-{model.replace(':','-')}"
