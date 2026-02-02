import pandas as pd
from .constants import DATASET_PATHS, COLUMN_RENAMES, COLUMN_DROP


def _load_raw_data(dataset_key: str, partition: str = "train"):
    """Loads raw data from specified dataset and partition.

    Args:
        dataset_key (str): Short alias of dataset used to retrieve file path.
            Example: 'AMGO', 'WAAM'
        partition (str): Data partition to load.
            Must be one of ['train', 'valid', 'test']

    Returns:
        tuple: Contains three dataframes:
            - tableA (pd.DataFrame): Source table
            - tableB (pd.DataFrame): Target table
            - df_partition (pd.DataFrame): Contains matching pairs labels between tableA and tableB
    """
    paths = DATASET_PATHS[dataset_key]

    table_a = pd.read_csv(paths["tableA"])
    table_b = pd.read_csv(paths["tableB"])

    partition_path = paths.get(partition)
    if partition_path is None:
        raise ValueError(f"No path found for partition '{partition}' in dataset '{dataset_key}'.")

    df_partition = pd.read_csv(partition_path)

    return table_a, table_b, df_partition


def prepare_structured_data(dataset_key: str, partition: str = "train") -> pd.DataFrame:
    """
    Processes raw data into a structured dataframe with matching pairs.

    Creates a merged dataframe where each row contains an instance from tableA,
    an instance from tableB, and their corresponding matching label.

    Args:
        dataset_key (str): Short alias of dataset used to retrieve file path.
            Example: 'AMGO', 'WAAM'
        partition (str): Data partition to load.
            Must be one of ['train', 'valid', 'test']

    Returns:
        pd.DataFrame: Merged dataframe containing:
            - Columns from tableA (suffixed with '_a')
            - Columns from tableB (suffixed with '_b')
            - 'label': Binary indicator for matching pairs (0 or 1)

    Example:
        >>> table_a, table_b, df_partition = prepare_structured_data('AMGO', 'train')
    """
    table_a, table_b, pairs = _load_raw_data(dataset_key, partition)

    # Rename columns if needed
    rename_map = COLUMN_RENAMES.get(dataset_key, {})
    if rename_map:
        table_a = table_a.rename(columns=rename_map)
        table_b = table_b.rename(columns=rename_map)

    # Drop unnecessary columns
    drop_cols = COLUMN_DROP.get(dataset_key, [])
    if drop_cols:
        table_a = table_a.drop(columns=[col for col in drop_cols if col in table_a.columns])
        table_b = table_b.drop(columns=[col for col in drop_cols if col in table_b.columns])

    table_a = table_a.add_suffix("_a")
    table_b = table_b.add_suffix("_b")

    # Avoid ID collision by renaming
    table_a = table_a.rename(columns={"id_a": "ltable_id"})
    table_b = table_b.rename(columns={"id_b": "rtable_id"})

    # Merge with partition data
    merged = pairs.merge(table_a, on="ltable_id", how="left")
    merged = merged.merge(table_b, on="rtable_id", how="left")

    return merged


def structured_to_text_data(structured_data: pd.DataFrame, with_semantic: bool = True) -> pd.DataFrame:
    """
    Converts structured data into text data.
    Args:
        structured_data (pd.DataFrame): Structured data with matching pairs.
            Must be a dataframe with columns 'id_a', 'id_b', 'label'.
        with_semantic (bool, optional): If True, the column names are serialized.
    Returns:
        pd.DataFrame: Text data with matching pairs.
    """
    def _serialize_entity(row, with_semantic) -> str:
        parts = []
        for col_name, value in row.items():
            text_value = str(value) if pd.notnull(value) else "nan"
            if with_semantic:
                parts.append(f"{col_name[:-2]}: {text_value}")
            else:
                parts.append(text_value)
        return "; ".join(parts)

    a_cols = [col for col in structured_data.columns if col.endswith("_a")]
    b_cols = [col for col in structured_data.columns if col.endswith("_b")]

    structured_data["entityA"] = structured_data[a_cols].apply(
        lambda row: _serialize_entity(row, with_semantic=with_semantic), axis=1)
    structured_data["entityB"] = structured_data[b_cols].apply(
        lambda row: _serialize_entity(row, with_semantic=with_semantic), axis=1)

    return structured_data[["ltable_id", "rtable_id", "entityA", "entityB", 'label']]


def entity_to_text(entity_row: pd.Series, with_semantic: bool = True) -> str:
    """
    Converts a single entity row to text format.
    
    Args:
        entity_row (pd.Series): Single row from entity DataFrame
        with_semantic (bool): If True, include column names in the output
    
    Returns:
        str: Serialized entity text
    """
    parts = []
    for col_name, value in entity_row.items():
        if col_name == 'id':  # Skip id column
            continue
        text_value = str(value) if pd.notnull(value) else "nan"
        if with_semantic:
            parts.append(f"{col_name}: {text_value}")
        else:
            parts.append(text_value)
    return "; ".join(parts)


def prepare_text_data(dataset_key: str, partition: str = "train", with_semantic: bool = True)-> pd.DataFrame:
    """
    Processes raw data into a dataframe with matching pairs in textual format.

    Creates a merged dataframe where each row contains a serialized instance from tableA,
    a serialized instance from tableB, and their corresponding matching label.

    Args:
        dataset_key (str): Short alias of dataset used to retrieve file path.
            Example: 'AMGO', 'WAAM'
        partition (str): Data partition to load.
            Must be one of ['train', 'valid', 'test']
        with_semantic (bool, optional): If True, the column names are serialized.

    Returns:
        pd.DataFrame: Merged dataframe containing:
            - serialized instances from tableA accessiable with column name 'entityA'
            - serialized instances from tableB accessiable with column name 'entityB'
            - 'label': Binary indicator for matching pairs (0 or 1)

    Example:
        >>> table_a, table_b, df_partition = prepare_text_data('AMGO', 'train')
    """
    structured_data = prepare_structured_data(dataset_key, partition)

    return structured_to_text_data(structured_data, with_semantic)


