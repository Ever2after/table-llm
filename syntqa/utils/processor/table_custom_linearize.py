# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatten sequence
"""
import abc
import abc
from typing import Dict, List, List
import pandas as pd


class TableLinearize(abc.ABC):

    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        df = pd.DataFrame(table_content['rows'], columns=table_content['header'])
        return df_to_table_prompt(df)
    
    def process_header(self, headers: List):
        lines = []
        lines.append('-- Columns:')
        for col in headers:
            lines.append(f'--   {col}')
        return '\n'.join(lines)
    
    def process_row(self, row: List, row_index: int):
        row_values = [str(val) for val in row]
        return f'--   {" | ".join(row_values)}'

def guess_sql_type(dtype) -> str:
    """
    Pandas dtype을 보고 SQL에서 자주 쓰이는 자료형으로 매핑해주는 함수 예시입니다.
    상황과 DBMS에 따라 원하는 자료형을 추가/조정하세요.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DECIMAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    # 날짜/시계열 타입 등 추가적으로 처리하려면 여기에 elif 추가
    else:
        # 문자열(object) 등은 TEXT로 처리
        return "TEXT"

def df_to_table_prompt(df: pd.DataFrame) -> str:
    """
    주어진 df(DataFrame)를 -- Table: ... 형태의 문자열로 변환합니다.
    """
    # 테이블 명 줄 만들기
    lines = []
    lines.append("-- Columns:")
    
    # 각 컬럼에 대한 자료형 매핑
    for col in df.columns:
        col_type = guess_sql_type(df[col].dtypes)
        # lines.append(f"--   {col} ({col_type})")
        lines.append(f"--   {col}")

    # 행(Row) 정보 구성
    lines.append("--")
    lines.append("-- Rows:")
    # 각 행을 출력 형식에 맞게 구성
    for idx, row in df.iterrows():
        # 예: "  Braden | 76"
        row_values = [str(val) for val in row.values]
        lines.append(f"--   {' | '.join(row_values)}")

    # 최종 문자열
    table_prompt = "\n".join(lines)
    return table_prompt