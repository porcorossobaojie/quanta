# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:59:58 2026

@author: Porco Rosso
"""
import inspect
import pandas as pd
import duckdb

from src.quanta.config import settings
from src.quanta.libs.db._engines.meta import main as meta
config = settings('libs').db.DuckDB
from src.quanta.libs.db._data_type_standard import data_trans



class main(meta, type('', (), config.recommand_settings)):

    @classmethod
    def __engine__(cls, **kwargs) -> duckdb.DuckDBPyConnection:
        """
        ===========================================================================

        Establishes a connection to the DuckDB database.

        Parameters
        ----------
        self : object
            The instance of the class.
        **kwargs : Any
            Keyword arguments for database connection parameters.

        Returns
        -------
        duckdb.DuckDBPyConnection
            A DuckDB database connection object.

        ---------------------------------------------------------------------------

        建立与 DuckDB 数据库的连接。

        参数
        ----------
        self : object
            类的实例。
        **kwargs : Any
            数据库连接参数的关键字参数。

        返回
        -------
        duckdb.DuckDBPyConnection
            一个 DuckDB 数据库连接对象。

        ---------------------------------------------------------------------------
        """

        database = f"{cls.path}/{cls.database}.duckdb"
        x = duckdb.connect(database=database, read_only = kwargs.get('read_only', False))
        x.execute(f'CREATE SCHEMA IF NOT EXISTS {cls.schema}')
        x.execute("SET wal_autocheckpoint='2GB';")
        return x

    @classmethod
    def __command__(cls, command: str, **kwargs: Any) -> pd.DataFrame:
         with cls.__engine__(**kwargs) as engine:
             x = engine.execute(command).fetchdf()
             return x

    @classmethod
    def __schema_info__(cls):
        info_params = {
            'schema': 'INFORMATION_SCHEMA',
            'table': 'COLUMNS',
            'columns': '*'
        }
        command = "SELECT {columns} FROM {schema}.{table}".format(**info_params)
        df = cls.__command__(command=command, read_only=True).iloc[:, [0,1,2,3,-1]]
        return df

    @classmethod
    def __find__(cls, col, df = None):
        df = cls.__schema_info__() if df is None else df
        mask_matrix = df.apply(lambda s: s.astype(str).str.contains(col, case=False, na=False))
        df = df[mask_matrix.any(axis=1)]
        return df

    @classmethod
    def __data_trans__(cls, str_obj):
        return data_trans(str_obj, recommand='DuckDB')


    def __columns_connect__(
        cls,
        columns_obj,
        type_position = 0,
        comment_position = 1,
    ):
        """
        ===========================================================================

        Connects and formats column information for DuckDB.

        This method takes a column object and formats it into a string suitable
        for SQL queries or a tuple containing column definitions and comments.

        Parameters
        ----------
        self : object
            The instance of the class.
        columns_obj : Optional[Union[str, List[str], Dict[str, Any]]]
            Column object. Can be:
            - None: Returns '*' for selecting all columns.
            - str: Returns the string as is.
            - list: Joins the list elements with ', ' for column selection.
            - dict: Formats dictionary keys and values into column definitions
              with types and comments.
        type_position : Optional[int], optional
            For dictionary `columns_obj`, the index of the type within the value list, by default None.
        comment_position : Optional[int], optional
            For dictionary `columns_obj`, the index of the comment within the value list, by default None.

        Returns
        -------
        Union[str, Tuple[str, Dict[str, str]]]
            Formatted column information. Returns a string if `columns_obj` is
            None, str, or list. Returns a tuple of (column_definitions_string, comments_dictionary)
            if `columns_obj` is a dictionary.

        ---------------------------------------------------------------------------

        连接并格式化 DuckDB 的列信息。

        此方法接受一个列对象，并将其格式化为适合 SQL 查询的字符串，
        或包含列定义和注释的元组。

        参数
        ----------
        self : object
            类的实例。
        columns_obj : Optional[Union[str, List[str], Dict[str, Any]]]
            列对象。可以是：
            - None：返回 '*' 以选择所有列。
            - str：直接返回字符串。
            - list：将列表元素用 ', ' 连接起来以供列选择。
            - dict：将字典的键和值格式化为包含类型和注释的列定义。
        type_position : Optional[int], optional
            对于字典 `columns_obj`，值列表中类型所在的索引，默认为 None。
        comment_position : Optional[int], optional
            对于字典 `columns_obj`，值列表中注释所在的索引，默认为 None。

        返回
        -------
        Union[str, Tuple[str, Dict[str, str]]]
            格式化的列信息。如果 `columns_obj` 为 None、str 或 list，则返回字符串。
            如果 `columns_obj` 为字典，则返回 (列定义字符串, 注释字典) 的元组。

        ---------------------------------------------------------------------------
        """
        if columns_obj is None:
            x = '*'
        elif isinstance(columns_obj, str):
            x = columns_obj
        elif isinstance(columns_obj, list):
            x = ', '.join([str(i) for i in columns_obj])
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
        return x

    def __read__(
        self,
        log: bool = False,
        show_time: bool = False,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Reads data from the DuckDB database.

        Parameters
        ----------
        self : object
            The instance of the class.
        log : bool, optional
            Whether to log the SQL command, by default False.
        show_time : bool, optional
            Whether to show the execution time, by default False.
        **kwargs : Any
            Additional keyword arguments for query parameters.

        Returns
        -------
        pd.DataFrame
            The read data as a pandas DataFrame.

        ---------------------------------------------------------------------------

        从 DuckDB 数据库读取数据。

        参数
        ----------
        self : object
            类的实例。
        log : bool, optional
            是否记录 SQL 命令，默认为 False。
        show_time : bool, optional
            是否显示执行时间，默认为 False。
        **kwargs : Any
            查询参数的额外关键字参数。

        返回
        -------
        pd.DataFrame
            读取的数据作为 pandas DataFrame。

        ---------------------------------------------------------------------------
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
            columns = parameters.get('columns', None)
            columns = list(columns.keys()) if isinstance(columns, dict) else columns
            columns = self.__columns_connect__(columns)

            parameters['columns'] = columns

            command = 'SELECT {columns} FROM {schema}.{table}'.format(
                **parameters
            )
            if parameters.get('where', None) is not None:
                where_clause = parameters['where'].replace('"', "'")
                command = f"{command} WHERE {where_clause}"

            if log:
                print(command)
            x = self.__command__(command, read_only=True, **parameters)
            return x
        return wraps_function()


