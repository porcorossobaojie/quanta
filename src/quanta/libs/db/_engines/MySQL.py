# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:59:58 2026

@author: Porco Rosso
"""
import inspect
from datetime import datetime
from  typing import Literal
import pandas as pd
from sqlalchemy import Engine, create_engine
import pymysql
from numpy import isreal

from quanta.config import settings
from quanta.libs.db._engines.meta import main as meta
config = settings('libs').db.MySQL
from quanta.libs.db._data_type_standard import data_trans



class main(meta, type('', (), config.recommand_settings)):
    @classmethod
    def __URL__(cls):
        return (
            f"{cls.mysql_connect}{cls.user}:{cls.password}@{cls.host}:{cls.port}/{cls.schema}?charset={cls.charset}"
        )
    
    @classmethod
    def __engine__(cls, **kwargs):
        connection_string = cls.__URL__()
        engine = create_engine(connection_string)
        return engine
    
    @classmethod
    def __env_init__(cls):
        cls.__command__(f'CREATE SCHEMA IF NOT EXISTS {cls.schema}')

    @classmethod
    def __command__(cls, command: str, **kwargs) -> pd.DataFrame:
        parameters = {i:config.recommand_settings[i] for i in ['host', 'port', 'user', 'password', 'charset', 'schema']}
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
    def __schema_info__(cls):
        info_params = {
            'schema': 'INFORMATION_SCHEMA',
            'table': 'COLUMNS',
            'columns': '*'
        }
        command = "SELECT {columns} FROM {schema}.{table}".format(**info_params)
        df = pd.read_sql(command, con=cls.__engine__())[['TABLE_SCHEMA', 'TABLE_NAME', 'COLUMN_NAME', 'COLUMN_COMMENT']]
        return df

    @classmethod
    def __find__(cls, col, df = None):
        df = cls.__schema_info__() if df is None else df
        mask_matrix = df.apply(lambda s: s.astype(str).str.contains(col, case=False, na=False))
        df = df[mask_matrix.any(axis=1)]
        return df

    @classmethod
    def __data_trans__(cls, str_obj):
        return data_trans(str_obj, recommand_settings='MySQL')


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
            x = ', \n'.join(
                [f'`{i}` {j[0]} DEFAULT NULL COMMENT "{j[1]}"' for i, j in x.items()]
            )
        return x

    def __read__(
        self,
        chunksize = None,
        log: bool = False,
        show_time: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        ===========================================================================

        Reads data from the MySQL database.

        Parameters
        ----------
        self : object
            The instance of the class.
        chunksize : Optional[int], optional
            Number of rows to read at a time, by default None.
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

        从 MySQL 数据库读取数据。

        参数
        ----------
        self : object
            类的实例。
        chunksize : Optional[int], optional
            每次读取的行数，默认为 None。
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
        schema = None,
        table = None,
        detail = False, 
        **kwargs
    ) -> bool:
        df = self.__schema_info__()
        if schema is not None:
            df = df[df['TABLE_SCHEMA'].lower() == schema.lower()]
        table = self.table if table is None else table
        df = df[df['TABLE_NAME'].lower() == table.lower()]
        if detail:
            return df
        else:
            return (True if len(df) else False)
        

    def __create_table__(
        self,
        primary_key = None,
        keys = None,
        partition = None,
        log: bool = False,
        **kwargs
    ) -> None:
        args = inspect.getargvalues(inspect.currentframe())
        args = {i: args.locals[i] for i in args.args if i != 'self'}
        parameters = self.__parameters__(args, kwargs)

        sql_command = 'CREATE TABLE IF NOT EXISTS `{schema}`.`{table}` (\n'.format(**parameters)
        if primary_key:
            if partition is None:
                primary_command = (
                    f'`{primary_key}` INT NOT NULL AUTO_INCREMENT PRIMARY KEY, '
                )
            else:
                partition_key = list(partition.keys())[0]
                primary_command = (
                    f'`{primary_key}` INT NOT NULL AUTO_INCREMENT, UNIQUE KEY (`{primary_key}`, `{partition_key}`), '
                )
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

            partition_parts = [
                f'PARTITION p{i} VALUES LESS THAN ({j})' for i, j in enumerate(partition_values_str)
                ]

            partition_command = key_part + ',\n'.join(partition_parts) + ')'
            sql_command += partition_command

        if log:
            print(sql_command)
        self.__command__(sql_command, **kwargs)
                
    def __drop_table__(self, log: bool = False, **kwargs) -> None:
        parameters = self.__parameters__(kwargs)
        sql_command = 'DROP TABLE IF EXISTS {schema}.{table}'.format(
            **parameters
        )
        self.__command__(sql_command, **kwargs)
        if log:
            print(sql_command)        
        
    def __write__(
        self,
        df_obj: pd.DataFrame,
        if_exists: Literal['fail', 'replace', 'append'] = 'append',
        index: bool = False,
        log: bool = False,
        **kwargs
    ):
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
            print(
                "Written DataFrame to <{schema}.{table}>: {count} records.".format(count=len(df_obj), **parameters)
            )
            
