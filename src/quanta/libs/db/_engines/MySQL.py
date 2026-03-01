# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:59:58 2026

@author: Porco Rosso
"""

import inspect
from datetime import datetime
from typing import Literal, Optional, Union, List, Dict, Any, Tuple
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from numpy import isreal

from quanta.config import settings
from quanta.libs.db._engines.meta import main as meta
from quanta.libs.db._data_type_standard import data_trans

config = settings('libs').db.MySQL


class main(meta, type('', (), config.recommand_settings)):
    @classmethod
    def __URL__(cls) -> str:
        """Constructs MySQL connection URL | 构建 MySQL 连接 URL"""
        return (
            f"{cls.mysql_connect}{cls.user}:{cls.password}@"
            f"{cls.host}:{cls.port}/{cls.schema}?charset={cls.charset}"
        )

    @classmethod
    def __engine__(cls, **kwargs: Any) -> Any:
        """Creates a SQLAlchemy engine | 创建 SQLAlchemy 引擎"""
        connection_string = cls.__URL__()
        engine = create_engine(connection_string)
        return engine

    @classmethod
    def __env_init__(cls) -> None:
        """Initializes database schema | 初始化数据库模式"""
        cls.__command__(f'CREATE SCHEMA IF NOT EXISTS {cls.schema}')

    @classmethod
    def __command__(cls, command: str, **kwargs: Any) -> Any:
        """
        =======================================================================
        Executes a raw SQL command using pymysql.

        Parameters
        ----------
        command : str
            The SQL command to execute.
        **kwargs : Any
            Override database connection parameters.

        Returns
        -------
        Any
            The result of fetchall().
        -----------------------------------------------------------------------
        使用 pymysql 执行原始 SQL 命令.

        参数
        ----
        command : str
            要执行的 SQL 命令.
        **kwargs : Any
            覆盖数据库连接参数.

        返回
        ----
        Any
            fetchall() 的结果.
        -----------------------------------------------------------------------
        """
        parameters = {i: config.recommand_settings[i] for i in ['host', 'port', 'user', 'password', 'charset', 'schema']}
        parameters.update(kwargs)
        parameters['db'] = parameters.pop('schema')

        con = pymysql.connect(**parameters)
        cur = con.cursor()
        cur.execute(command)
        x = cur.fetchall()
        con.commit()
        cur.close()
        con.close()
        return x

    @classmethod
    def __schema_info__(cls) -> pd.DataFrame:
        """Retrieves column metadata from INFORMATION_SCHEMA | 获取列元数据"""
        info_params = {
            'schema': 'INFORMATION_SCHEMA',
            'table': 'COLUMNS',
            'columns': '*'
        }
        command = "SELECT {columns} FROM {schema}.{table}".format(**info_params)
        df = pd.read_sql(command, con=cls.__engine__())[['TABLE_SCHEMA', 'TABLE_NAME', 'COLUMN_NAME', 'COLUMN_COMMENT']]
        return df

    @classmethod
    def __find__(cls, col: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Searches for columns in schema metadata | 在模式元数据中搜索列"""
        df = cls.__schema_info__() if df is None else df
        mask_matrix = df.apply(lambda s: s.astype(str).str.contains(col, case=False, na=False))
        df = df[mask_matrix.any(axis=1)]
        return df

    @classmethod
    def __data_trans__(cls, str_obj: str) -> str:
        """Translates data types to MySQL standard | 将数据类型转换为 MySQL 标准"""
        return data_trans(str_obj, recommand_settings='MySQL')

    def __columns_connect__(
        cls,
        columns_obj: Optional[Union[str, List[str], Dict[str, Any]]],
        type_position: int = 0,
        comment_position: int = 1,
    ) -> str:
        """
        =======================================================================
        Connects and formats column information for MySQL SQL queries.

        Parameters
        ----------
        columns_obj : Optional[Union[str, List[str], Dict[str, Any]]]
            Column object (None, str, list, or dict).
        type_position : int, optional
            Index of type in dict values, by default 0.
        comment_position : int, optional
            Index of comment in dict values, by default 1.

        Returns
        -------
        str
            Formatted SQL column string.
        -----------------------------------------------------------------------
        连接并格式化用于 MySQL SQL 查询的列信息.

        参数
        ----
        columns_obj : Optional[Union[str, List[str], Dict[str, Any]]]
            列对象 (None, 字符串, 列表或字典).
        type_position : int, optional
            字典值中类型的索引, 默认为 0.
        comment_position : int, optional
            字典值中注释的索引, 默认为 1.

        返回
        ----
        str
            格式化后的 SQL 列字符串.
        -----------------------------------------------------------------------
        """
        if columns_obj is None:
            x = '*'
        elif isinstance(columns_obj, str):
            x = columns_obj
        elif isinstance(columns_obj, list):
            x = ', '.join([str(i) for i in columns_obj])
        elif isinstance(columns_obj, dict):
            formatted = {}
            for i, j in columns_obj.items():
                formatted[i] = [
                    cls.__data_trans__(j[type_position].upper()),
                    '' if len(j) == 1 else j[comment_position]
                ]
            x = ', \n'.join(
                [f'`{i}` {j[0]} DEFAULT NULL COMMENT "{j[1]}"' for i, j in formatted.items()]
            )
        return x

    def __read__(
        self,
        chunksize: Optional[int] = None,
        log: bool = False,
        show_time: bool = False,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        =======================================================================
        Reads data from the MySQL database with optional chunking.

        Parameters
        ----------
        chunksize : int, optional
            Number of rows to read per chunk, by default None.
        log : bool, optional
            Whether to print SQL command, by default False.
        show_time : bool, optional
            Whether to show execution time, by default False.
        **kwargs : Any
            Additional query parameters.

        Returns
        -------
        pd.DataFrame
            The read data as a pandas DataFrame.
        -----------------------------------------------------------------------
        从 MySQL 数据库读取数据, 可选分块读取.

        参数
        ----
        chunksize : int, optional
            每块读取的行数, 默认为 None.
        log : bool, optional
            是否打印 SQL 命令, 默认为 False.
        show_time : bool, optional
            是否显示执行时间, 默认为 False.
        **kwargs : Any
            额外的查询参数.

        返回
        ----
        pd.DataFrame
            读取的数据作为 pandas DataFrame.
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
            columns = parameters.get('columns', None)
            columns = list(columns.keys()) if isinstance(columns, dict) else columns
            columns = self.__columns_connect__(columns)

            parameters['columns'] = columns

            sql_command = 'SELECT {columns} FROM {schema}.{table}'.format(**parameters)
            if parameters.get('where', None) is not None:
                sql_command = '{} WHERE {where}'.format(sql_command, **parameters)

            if log:
                print(sql_command)

            engine = self.__engine__(**parameters)

            if parameters.get('chunksize', None) is not None:
                offset = 0
                chunks = []
                while True:
                    order_params = {
                        **parameters,
                        'offset': offset,
                        'sql_command': sql_command
                    }
                    order = ('{sql_command} LIMIT {chunksize} OFFSET {offset}').format_map(order_params)
                    obj = pd.read_sql(order, con=engine)
                    chunks.append(obj)
                    if len(obj) < parameters.get('chunksize', 1):
                        break
                    else:
                        offset += parameters.get('chunksize', 1)
                return pd.concat(chunks)
            else:
                return pd.read_sql(sql_command, con=engine)

        return wraps_function()

    def __table_exist__(
        self,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        detail: bool = False,
        **kwargs: Any
    ) -> Union[bool, pd.DataFrame]:
        """Checks if a table exists in MySQL | 检查 MySQL 中是否存在指定表"""
        df = self.__schema_info__()
        if schema is not None:
            df = df[df['TABLE_SCHEMA'].lower() == schema.lower()]
        table = getattr(self, 'table', None) if table is None else table
        df = df[df['TABLE_NAME'].lower() == table.lower()]
        return df if detail else (len(df) > 0)

    def __create_table__(
        self,
        primary_key: Optional[str] = None,
        keys: Optional[Union[str, List[str]]] = None,
        partition: Optional[Dict[str, List[Any]]] = None,
        log: bool = False,
        **kwargs: Any
    ) -> None:
        """
        =======================================================================
        Creates a table in MySQL with optional indexing and partitioning.

        Parameters
        ----------
        primary_key : str, optional
            Column name for the primary key, by default None.
        keys : Union[str, List[str]], optional
            Columns to index, by default None.
        partition : Dict[str, List[Any]], optional
            Partition rules, by default None.
        log : bool, optional
            Whether to print SQL command, by default False.
        **kwargs : Any
            Additional schema and table parameters.
        -----------------------------------------------------------------------
        在 MySQL 中创建表, 可选索引和分区设置.

        参数
        ----
        primary_key : str, optional
            主键列名, 默认为 None.
        keys : Union[str, List[str]], optional
            要建立索引的列, 默认为 None.
        partition : Dict[str, List[Any]], optional
            分区规则, 默认为 None.
        log : bool, optional
            是否打印 SQL 命令, 默认为 False.
        **kwargs : Any
            额外的模式和表参数.
        -----------------------------------------------------------------------
        """
        args = inspect.getargvalues(inspect.currentframe())
        args = {i: args.locals[i] for i in args.args if i != 'self'}
        parameters = self.__parameters__(args, kwargs)

        sql_command = 'CREATE TABLE IF NOT EXISTS `{schema}`.`{table}` (\n'.format(**parameters)
        if primary_key:
            if partition is None:
                primary_command = f'`{primary_key}` INT NOT NULL AUTO_INCREMENT PRIMARY KEY, '
            else:
                partition_key = list(partition.keys())[0]
                primary_command = f'`{primary_key}` INT NOT NULL AUTO_INCREMENT, UNIQUE KEY (`{primary_key}`, `{partition_key}`), '
            sql_command += primary_command

        columns_text = self.__columns_connect__(parameters.get('columns', {}))
        sql_command += columns_text

        if keys:
            keys_command = ',\n' + ',\n'.join([f'key ({i})' if isinstance(i, str) else f'key({",".join(i)})' for i in ([keys] if isinstance(keys, str) else keys)])
            sql_command += keys_command

        char_col_command = (
            ')\n ENGINE = InnoDB DEFAULT CHARSET = {charset} COLLATE = {collate}'
        ).format(**parameters)
        sql_command += char_col_command

        if partition:
            part_key = list(partition.keys())[0]
            key_part = f'\n PARTITION BY RANGE COLUMNS(`{part_key}`)(\n'

            part_values = list(partition.values())[0]
            partition_values_str = []
            for val in part_values:
                if not isreal(val) or isinstance(val, datetime):
                    partition_values_str.append(f'"{val}"')
                else:
                    partition_values_str.append(str(val))
            partition_values_str.append('MAXVALUE')

            partition_parts = [f'PARTITION p{i} VALUES LESS THAN ({j})' for i, j in enumerate(partition_values_str)]
            partition_command = key_part + ',\n'.join(partition_parts) + ')'
            sql_command += partition_command

        if log:
            print(sql_command)
        self.__command__(sql_command, **kwargs)

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
        Writes a pandas DataFrame to the MySQL database.

        Parameters
        ----------
        df_obj : pd.DataFrame
            The DataFrame to be written.
        if_exists : Literal['fail', 'replace', 'append'], optional
            Action to take if table exists, by default 'append'.
        index : bool, optional
            Whether to write index, by default False.
        log : bool, optional
            Whether to print status, by default False.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        None
        -----------------------------------------------------------------------
        将 pandas DataFrame 写入 MySQL 数据库.

        参数
        ----
        df_obj : pd.DataFrame
            要写入的 DataFrame.
        if_exists : Literal['fail', 'replace', 'append'], optional
            如果表已存在时的操作, 默认为 'append'.
        index : bool, optional
            是否写入索引, 默认为 False.
        log : bool, optional
            是否打印状态, 默认为 False.
        **kwargs : Any
            额外参数.

        返回
        ----
        None
        -----------------------------------------------------------------------
        """
        parameters = self.__parameters__(kwargs)
        engine = self.__engine__(**parameters)
        df_obj.to_sql(
            parameters['table'],
            con=engine,
            if_exists=if_exists,
            index=index,
            chunksize=320000
        )
        engine.dispose()
        if log:
            print("Written DataFrame to <{schema}.{table}>: {count} records.".format(count=len(df_obj), **parameters))
