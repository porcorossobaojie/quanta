# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:59:58 2026

@author: Porco Rosso
"""

import inspect
from typing import Literal, Optional, Union, List, Dict, Any, Tuple
import pandas as pd
from pathlib import Path
import duckdb

from ....config import settings
from .meta import main as meta
from ...db._data_type_standard import data_trans

config = settings('libs').db.DuckDB


class main(meta, type('', (), config.recommand_settings)):
    """
    ===========================================================================
    DuckDB database engine implementation.
    ---------------------------------------------------------------------------
    DuckDB 数据库引擎实现.
    ---------------------------------------------------------------------------
    """

    @classmethod
    def __setting__ (cls, **kwargs: Any) -> None:
        """Updates class settings with keyword arguments | 用关键字参数更新类设置"""
        [setattr(cls, i, j) for i,j in kwargs.items()]

    @classmethod
    def __reset__ (cls) -> None:
        """Resets class settings to configured defaults | 将类设置重置为配置的默认值"""
        [setattr(cls, i, j) for i,j in  config.recommand_settings.items()]

    @classmethod
    def __engine__(cls, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        """
        =======================================================================
        Establishes a connection to the DuckDB database.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments for database connection parameters, e.g.,
            'read_only'.

        Returns
        -------
        duckdb.DuckDBPyConnection
            A DuckDB database connection object.
        -----------------------------------------------------------------------
        建立与 DuckDB 数据库的连接.

        参数
        ----
        **kwargs : Any
            数据库连接参数的关键字参数, 例如 'read_only'.

        返回
        ----
        duckdb.DuckDBPyConnection
            一个 DuckDB 数据库连接对象.
        -----------------------------------------------------------------------
        """
        path = kwargs.get('path', cls.path)
        database = kwargs.get('database', cls.database)
        database = f"{path}/{database}.duckdb"
        x = duckdb.connect(database=database, read_only=kwargs.get('read_only', False))
        x.execute("SET wal_autocheckpoint='2GB';")
        return x

    @classmethod
    def __env_init__(cls) -> None:
        """Initializes database schema | 初始化数据库模式"""
        cls.__command__(f'CREATE SCHEMA IF NOT EXISTS {cls.schema}', read_only=False)
        (Path(cls.path) / cls.parquet).mkdir(parents=True, exist_ok=True)

    @classmethod
    def __command__(cls, command: str, **kwargs: Any) -> pd.DataFrame:
        """Executes a SQL command and returns the result | 执行 SQL 命令并返回结果"""
        with cls.__engine__(**kwargs) as engine:
            x = engine.execute(command).fetchdf()
            return x

    @classmethod
    def __schema_info__(cls) -> pd.DataFrame:
        """Retrieves schema information from metadata | 从元数据获取模式信息"""
        info_params = {
            'schema': 'INFORMATION_SCHEMA',
            'table': 'COLUMNS',
            'columns': '*'
        }
        command = "SELECT {columns} FROM {schema}.{table}".format(**info_params)
        df = cls.__command__(command=command, read_only=True).iloc[:, [0, 1, 2, 3, -1]]
        return df

    @classmethod
    def __find__(cls, col: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Searches for columns in the schema metadata | 在模式元数据中搜索列"""
        df = cls.__schema_info__() if df is None else df
        mask_matrix = df.apply(lambda s: s.astype(str).str.contains(col, case=False, na=False))
        df = df[mask_matrix.any(axis=1)]
        return df

    @classmethod
    def __data_trans__(cls, str_obj: str) -> str:
        """Translates data types to DuckDB standard | 将数据类型转换为 DuckDB 标准"""
        return data_trans(str_obj, recommand_settings='DuckDB')

    def __columns_connect__(
        cls,
        columns_obj: Optional[Union[str, List[str], Dict[str, Any]]],
        type_position: int = 0,
        comment_position: int = 1,
    ) -> Union[str, Tuple[str, Dict[str, str]]]:
        """
        =======================================================================
        Connects and formats column information for DuckDB SQL queries.

        Parameters
        ----------
        columns_obj : Optional[Union[str, List[str], Dict[str, Any]]]
            Column object. Can be None (select all), a string, a list, or a
            dictionary with types and comments.
        type_position : int, optional
            Index of the data type in dictionary values, by default 0.
        comment_position : int, optional
            Index of the comment in dictionary values, by default 1.

        Returns
        -------
        Union[str, Tuple[str, Dict[str, str]]]
            Formatted column string or a tuple of (SQL string, comment dict).
        -----------------------------------------------------------------------
        连接并格式化用于 DuckDB SQL 查询的列信息.

        参数
        ----
        columns_obj : Optional[Union[str, List[str], Dict[str, Any]]]
            列对象. 可以是 None (全选), 字符串, 列表或包含类型和注释的字典.
        type_position : int, optional
            字典值中数据类型的索引位置, 默认为 0.
        comment_position : int, optional
            字典值中注释的索引位置, 默认为 1.

        返回
        ----
        Union[str, Tuple[str, Dict[str, str]]]
            格式化后的列字符串, 或 (SQL 字符串, 注释字典) 元组.
        -----------------------------------------------------------------------
        """
        if columns_obj is None:
            x = '*'
        elif isinstance(columns_obj, str):
            x = columns_obj

        elif isinstance(columns_obj, dict):
            type_position = 0 if type_position is None else type_position
            comment_position = 1 if comment_position is None else comment_position

            x = {}
            for i, j in columns_obj.items():
                x[i] = [
                    cls.__data_trans__(j[type_position].upper()),
                    '' if len(j) == 1 else j[comment_position]
                ]
            x = (
                ', \n'.join([f'{i} {j[0]} DEFAULT NULL' for i, j in x.items()]),
                {i: j[1] for i, j in x.items()}
            )
        else:
            x = ', '.join([str(i) for i in columns_obj])
        return x

    def __read__(
        self,
        log: bool = False,
        show_time: bool = False,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        =======================================================================
        Reads data from the DuckDB database based on instance attributes.

        Parameters
        ----------
        log : bool, optional
            Whether to print the SQL command, by default False.
        show_time : bool, optional
            Whether to show execution time, by default False.
        **kwargs : Any
            Additional query parameters.

        Returns
        -------
        pd.DataFrame
            The resulting dataset as a pandas DataFrame.
        -----------------------------------------------------------------------
        根据实例属性从 DuckDB 数据库读取数据.

        参数
        ----
        log : bool, optional
            是否打印 SQL 命令, 默认为 False.
        show_time : bool, optional
            是否显示执行时间, 默认为 False.
        **kwargs : Any
            额外的查询参数.

        返回
        ----
        pd.DataFrame
            作为 pandas DataFrame 的查询结果集.
        -----------------------------------------------------------------------
        """
        args = inspect.getargvalues(inspect.currentframe())
        args = {i: args.locals[i] for i in args.args if i != 'self'}
        parameters = self.__parameters__(args, kwargs)
        schema = kwargs.get('schema', getattr(self, 'schema', None))
        table = kwargs.get('table', getattr(self, 'table', None))

        @self.__timing_decorator__(
            schema=schema, table=table, show_time=show_time
        )
        def wraps_function() -> pd.DataFrame:
            """Executes the timed read command | 执行带计时的读取命令"""
            columns = parameters.get('columns', None)
            columns = list(columns.keys()) if isinstance(columns, dict) else columns
            columns = self.__columns_connect__(columns)

            parameters['columns'] = columns

            command = 'SELECT {columns} FROM {schema}.{table}'.format(**parameters)
            if parameters.get('where', None) is not None:
                where_clause = parameters['where'].replace('"', "'")
                command = f"{command} WHERE {where_clause}"

            if log:
                print(command)
            x = self.__command__(command, read_only=True, **parameters)
            return x

        return wraps_function()

    @classmethod
    def __read_parquet__(cls, path, file_name=None):
        """
        =======================================================================
        Reads a parquet file into a DataFrame via DuckDB.

        Parameters
        ----------
        path : str
            The directory or file path of the parquet data.
        file_name : str, optional
            The parquet file name; when provided, path is treated as a
            directory. Default is None.

        Returns
        -------
        pd.DataFrame
            The DataFrame loaded from the parquet file.
        -----------------------------------------------------------------------
        通过 DuckDB 将 parquet 文件读取为 DataFrame.

        参数
        ----
        path : str
            parquet 数据的目录或文件路径.
        file_name : str, optional
            parquet 文件名; 提供时 path 被视为目录. 默认 None.

        返回
        ----
        pd.DataFrame
            从 parquet 文件加载的 DataFrame.
        -----------------------------------------------------------------------
        """
        if file_name is not None:
            if file_name.split('.')[-1] != 'parquet':
                file_name = f"{file_name}.parquet"
            command = f"SELECT * FROM READ_PARQUET('{path}/{file_name}')"
        else:
            command = f"SELECT * FROM READ_PARQUET('{path}')"
        x = cls.__command__(command, read_only=True)
        return x

    @classmethod
    def __write_parquet__(cls, df, path, file_name=None, log=True):
        """
        =======================================================================
        Writes a DataFrame to a parquet file under the given path.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to persist.
        path : str
            The destination directory for the parquet file.
        file_name : str, optional
            The parquet file name; defaults to 'data_0' when None.
        log : bool
            Whether to print a write summary. Default is True.
        -----------------------------------------------------------------------
        将 DataFrame 写入指定路径下的 parquet 文件.

        参数
        ----
        df : pd.DataFrame
            需要持久化的 DataFrame.
        path : str
            parquet 文件的输出目录.
        file_name : str, optional
            parquet 文件名; 为 None 时默认 'data_0'.
        log : bool
            是否打印写入摘要. 默认 True.
        -----------------------------------------------------------------------
        """
        file_name = 'data_0' if file_name is None else file_name
        if file_name.split('.')[-1] != 'parquet':
            file_name = f"{file_name}.parquet"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with cls.__engine__() as con:
            con.register('df_obj', df)
            con.execute(f"COPY (SELECT * FROM 'df_obj') TO '{path}/{file_name}' (FORMAT parquet, OVERWRITE_OR_IGNORE true)")
            con.close()
        if log:
            print(f"Written DataFrame to <{path}.{file_name}>: {df.shape[0]} records.")

    def __table_exist__(
        self,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        detail: bool = False,
        **kwargs: Any
    ) -> Union[bool, pd.DataFrame]:
        """Checks if a table exists in the schema | 检查模式中是否存在指定表"""
        df = self.__schema_info__()
        if schema is not None:
            df = df[df['table_schema'] == schema]
        table = self.table if table is None else table
        df = df[df['table_name'] == table]
        if detail:
            return df
        else:
            return (True if len(df) else False)

    def __create_table__(self, log: bool = False, **kwargs: Any) -> None:
        """Creates a table in DuckDB with optional column comments | 在 DuckDB 中创建表"""
        args = inspect.getargvalues(inspect.currentframe())
        args = {i: args.locals[i] for i in args.args if i != 'self'}
        parameters = self.__parameters__(args, kwargs)

        command = 'CREATE TABLE IF NOT EXISTS {schema}.{table} (\n'.format(**parameters)
        columns = self.__columns_connect__(parameters.get('columns', {}))
        if isinstance(columns, tuple):
            columns_str, comments_dict = columns
            command = command + columns_str + '\n);'
            comment_commands = [
                "COMMENT ON COLUMN {schema}.{table}.{i} IS '{j}';".format(
                    i=i, j=j, **parameters
                ) for i, j in comments_dict.items()
            ]
            command_list = [
                'BEGIN TRANSACTION;',
                command,
                *comment_commands,
                'COMMIT;'
            ]
            full_command = '\n'.join(command_list)
            self.__command__(full_command)
            if log:
                print(full_command)
        else:
            command = command + columns + '\n);'
            self.__command__(command)
            if log:
                print(command)

    def __drop_table__(self, log: bool = False, **kwargs: Any) -> None:
        """Drops a table if it exists | 如果存在则删除表"""
        parameters = self.__parameters__(kwargs)
        sql_command = 'DROP TABLE IF EXISTS {schema}.{table}'.format(**parameters)
        self.__command__(sql_command, **kwargs)
        if log:
            print(sql_command)

    def __write__(
        self,
        df_obj: pd.DataFrame,
        if_exists: Literal['fail', 'replace', 'append'] = 'append',
        index: bool = False,
        log: bool = False,
        **kwargs: Any
    ) -> None:
        """
        =======================================================================
        Writes a pandas DataFrame to the DuckDB database.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The DataFrame to be written.
        if_exists : Literal['fail', 'replace', 'append'], optional
            Action to take if table exists, by default 'append'.
        index : bool, optional
            Whether to write the DataFrame index, by default False.
        log : bool, optional
            Whether to print the write status, by default False.
        **kwargs : Any
            Additional parameters for schema and table selection.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        将 pandas DataFrame 写入 DuckDB 数据库.

        参数
        ----
        df_obj : pd.DataFrame
            要写入的 DataFrame.
        if_exists : Literal['fail', 'replace', 'append'], optional
            如果表已存在时的操作, 默认为 'append'.
        index : bool, optional
            是否写入 DataFrame 索引, 默认为 False.
        log : bool, optional
            是否打印写入状态, 默认为 False.
        **kwargs : Any
            用于模式和表选择的额外参数.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        if index:
            df_obj = df_obj.reset_index()

        parameters = self.__parameters__(kwargs)
        table_exist = self.__table_exist__(**parameters)

        with self.__engine__(**parameters) as con:
            con.register('df_obj', df_obj)

            insert_stmt = "INSERT INTO {database}.{schema}.{table} BY NAME SELECT * FROM df_obj".format(**parameters)
            create_stmt = "CREATE OR REPLACE TABLE {database}.{schema}.{table} AS SELECT * FROM df_obj".format(**parameters)

            if table_exist:
                if if_exists == 'replace':
                    self.__drop_table__(**parameters)
                    try:
                        self.__create_table__(**parameters)
                        con.execute(insert_stmt)
                    except Exception:
                        con.execute(create_stmt)
                elif if_exists == 'append':
                    con.execute(insert_stmt)
                elif if_exists == 'fail':
                    raise ValueError('Table already existed.')
                else:
                    raise ValueError("if_exists must be in ['fail', 'replace', 'append']")
            else:
                try:
                    self.__create_table__(**parameters)
                    con.execute(insert_stmt)
                except Exception:
                    con.execute(create_stmt)

            con.close()
        if log:
            print("Written DataFrame to <{schema}.{table}>: {count} records.".format(count=len(df_obj), **parameters))
