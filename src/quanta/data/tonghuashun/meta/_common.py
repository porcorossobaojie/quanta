# -*- coding: utf-8 -*-

import hashlib
import html
import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests

from quanta.config import settings
from quanta.libs.utils import merge_dicts

config = settings('data')

list_pattern = re.compile(
    r'id="report_list_contents"[^>]*>(.*?)</div>',
    re.DOTALL
)
report_id_pattern = re.compile(r'/(\d+)\.shtml(?:\?.*)?$')
script_pattern = re.compile(r'<script.*?</script>', re.DOTALL | re.IGNORECASE)
style_pattern = re.compile(r'<style.*?</style>', re.DOTALL | re.IGNORECASE)
tag_pattern = re.compile(r'<[^>]+>')
space_pattern = re.compile(r'[ \t\u3000]+')
blank_line_pattern = re.compile(r'\n\s*\n+')
rating_rules = [
    (5, ['强烈推荐', '强烈买入', '买入', '强推', '推荐', '优于大市',
         '跑赢行业', 'outperform', '审慎推荐']),
    (4, ['增持', '谨慎增持', '审慎增持', '优于行业', '超配', '看多']),
    (3, ['中性', '持有', '同步大市', '标配', '观望', '区间震荡']),
    (2, ['减持', '弱于大市', '低配', '回避']),
    (1, ['卖出', '强烈卖出'])
]
rating_names = {
    '强烈推荐': '买入', '强烈买入': '买入', '买入': '买入',
    '强推': '买入', '推荐': '买入', '优于大市': '买入',
    '跑赢行业': '买入', 'outperform': '买入', '审慎推荐': '买入',
    '增持': '增持', '谨慎增持': '增持', '审慎增持': '增持',
    '优于行业': '增持', '超配': '增持', '看多': '增持',
    '中性': '中性', '持有': '中性', '同步大市': '中性',
    '标配': '中性', '观望': '中性', '区间震荡': '中性',
    '减持': '减持', '弱于大市': '减持', '低配': '减持',
    '回避': '减持', '卖出': '卖出', '强烈卖出': '卖出'
}
target_price_patterns = [
    r'目标价(?:为|至|到)?\s*([\d]+(?:\.\d+)?)\s*元',
    r'([\d]+(?:\.\d+)?)\s*元(?:的)?目标价',
    r'(?:合理价值|合理估值|目标市值对应价格)(?:为|至)?\s*'
    r'([\d]+(?:\.\d+)?)\s*元',
    r'(?:6个月|六个月|12个月|一年)目标价(?:为|至)?\s*'
    r'([\d]+(?:\.\d+)?)\s*元',
    r'(?:上调|下调|调整)(?:公司)?目标价(?:至|为|到)\s*'
    r'([\d]+(?:\.\d+)?)\s*元'
]


class common:
    """
    ===========================================================================
    Shared metadata, HTTP, parsing, and pipeline hooks for Tonghuashun data.
    ---------------------------------------------------------------------------
    同花顺数据共用的元数据, HTTP, 解析及流水线钩子.
    ---------------------------------------------------------------------------
    """

    @property
    def portfolio_type(self) -> str:
        """Retrieves the portfolio type from the table name | 从表名获取资产类型"""
        for portfolio_type in config.public_keys.recommend_settings.portfolio_types:
            if portfolio_type in self.table:
                return portfolio_type
        return 'other'

    @property
    def code(self) -> str:
        """Retrieves the standard asset-code column | 获取标准资产代码字段"""
        return getattr(self, f'{self.portfolio_type}_code')

    @property
    def columns(self) -> Dict[str, List[str]]:
        """Builds local columns from configured source mappings | 从源字段映射生成本地字段"""
        return merge_dicts(*list(self.columns_information.values()))

    def __columns_rename__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames source columns to local standard columns | 将源字段重命名为本地标准字段"""
        rename_dict = {
            source: list(target.keys())[0]
            for source, target in self.columns_information.items()
        }
        df = df.rename(rename_dict, axis=1)
        return df.reindex(columns=list(self.columns.keys()))

    @staticmethod
    def __source_code__(code: str) -> str:
        """Gets the six-digit source code | 获取六位源代码"""
        source_code = str(code).strip().upper().split('.')[0]
        if not re.fullmatch(r'\d{6}', source_code):
            raise ValueError(f'Invalid stock code: {code}')
        return source_code

    @staticmethod
    def __standard_code__(code: str) -> str:
        """Normalizes code to quanta format | 标准化为 quanta 代码"""
        source_code = common.__source_code__(code)
        exchange = 'XSHG' if source_code.startswith(('5', '6', '9')) else 'XSHE'
        return f'{source_code}.{exchange}'

    @staticmethod
    def __report_id__(url: str) -> int:
        """Extracts a stable report ID from its URL | 从 URL 提取稳定研报 ID"""
        match = report_id_pattern.search(str(url).strip())
        if match is None:
            raise ValueError(f'Invalid report URL: {url}')
        return int(match.group(1))

    @staticmethod
    def __plain_text__(raw_html: str) -> str:
        """Converts report HTML into normalized plain text | 将研报 HTML 转换为标准纯文本"""
        body_position = raw_html.lower().find('<body')
        body = raw_html[body_position:] if body_position >= 0 else raw_html
        body = script_pattern.sub(' ', body)
        body = style_pattern.sub(' ', body)
        body = tag_pattern.sub(' ', body)
        body = html.unescape(body)
        body = space_pattern.sub(' ', body)
        return blank_line_pattern.sub('\n', body).strip()

    @staticmethod
    def __normalize_rating__(
        rating_raw: str,
        report_text: str
    ) -> Tuple[str, int]:
        """Normalizes vendor ratings into a name and score | 将供应商评级标准化为名称和评分"""
        candidate = str(rating_raw or '').strip()
        if not candidate:
            match = re.search(
                r'(?:维持|给予|首次覆盖|上调至|下调至)?\s*["“\']?'
                r'(强烈推荐|买入|增持|中性|持有|减持|卖出|推荐|'
                r'优于大市|跑赢行业)',
                report_text
            )
            candidate = match.group(1) if match else ''

        candidate_lower = candidate.lower()
        for score, names in rating_rules:
            for name in names:
                if name in candidate_lower:
                    return rating_names.get(name, name), score
        return candidate or '未评级', 0

    @staticmethod
    def __target_price__(report_text: str) -> Optional[float]:
        """Extracts the first valid target price | 提取首个有效目标价"""
        for pattern in target_price_patterns:
            values = [float(value) for value in re.findall(pattern, report_text)]
            values = [value for value in values if 0.5 <= value <= 100000]
            if values:
                return values[0]
        return None

    @staticmethod
    def __previous_target_price__(report_text: str) -> Optional[float]:
        """Extracts the previous target price | 提取前目标价"""
        patterns = [
            r'(?:目标价|合理价值)[^。;；]{0,30}?（前值\s*([\d.]+)\s*元',
            r'前值\s*([\d.]+)\s*元'
        ]
        for pattern in patterns:
            match = re.search(pattern, report_text)
            if match:
                value = float(match.group(1))
                if 0.5 <= value <= 100000:
                    return value
        return None

    @staticmethod
    def __revision__(report_text: str) -> str:
        """Extracts the forecast revision direction | 提取预测调整方向"""
        if re.search(r'首次覆盖|首次给予|首次评级|首次深度|首次推荐', report_text):
            return '首次'
        forecast = (
            r'(盈利预测|业绩预测|预测值|净利|净利润|EPS|每股收益|'
            r'每股盈利|营收预测|收入预测)'
        )
        if re.search(r'(下调|调低|下修)[^。;；]{0,15}' + forecast, report_text):
            return '下调'
        if re.search(r'(上调|调高|上修)[^。;；]{0,15}' + forecast, report_text):
            return '上调'
        if re.search(r'维持[^。;；]{0,20}(盈利预测|业绩预测|预测)', report_text):
            return '维持'
        return ''

    @staticmethod
    def __risk__(report_text: str) -> str:
        """Extracts the report risk disclosure | 提取研报风险提示"""
        match = re.search(
            r'(?:风险提示|风险因素|评级面临的主要风险)[:：]?\s*'
            r'([^。]{5,120})',
            report_text
        )
        return match.group(1).strip() if match else ''

    def __request__(self, url: str) -> str:
        """Requests and decodes a GBK source page | 请求并解码 GBK 源页面"""
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            if self._last_request_at is not None:
                elapsed = time.monotonic() - self._last_request_at
                delay = random.uniform(self.sleep_min, self.sleep_max)
                time.sleep(max(0, delay - elapsed))
            try:
                self._last_request_at = time.monotonic()
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.content.decode('gbk', errors='replace')
            except requests.RequestException as error:
                last_error = error
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f'Request failed <{url}>: {last_error}')

    def __report_list__(self, code: str) -> List[Dict[str, Any]]:
        """Fetches the complete report list of one stock | 获取单只股票的完整研报列表"""
        source_code = self.__source_code__(code)
        raw_html = self.__request__(self.list_url.format(code=source_code))
        match = list_pattern.search(raw_html)
        if match is None:
            raise ValueError(f'{source_code}: report_list_contents not found.')
        rows = json.loads(match.group(1))
        for row in rows:
            row['url'] = str(row.get('url', '')).replace('http://', 'https://')
            row['code'] = source_code
        return rows

    def __report_text__(self, url: str) -> str:
        """Fetches and cleans one public report page | 获取并清洗单篇公开研报页面"""
        return self.__plain_text__(self.__request__(url))

    def __raw_record__(
        self,
        row: Dict[str, Any],
        report_text: str,
        crawl_dt: pd.Timestamp
    ) -> Dict[str, Any]:
        """Builds one source-shaped report record | 构造单条源格式研报记录"""
        rating, rating_score = self.__normalize_rating__(
            str(row.get('thspj') or ''),
            report_text
        )
        return {
            'id': self.__report_id__(row['url']),
            'code': row['code'],
            'date': row.get('date'),
            'broker': str(row.get('source') or '').strip(),
            'analyst': str(row.get('researcher') or '').strip(),
            'title': str(row.get('title') or '').strip(),
            'rating_raw': str(row.get('thspj') or '').strip(),
            'rating': rating,
            'rating_score': rating_score,
            'target_price': self.__target_price__(report_text),
            'previous_target_price': self.__previous_target_price__(report_text),
            'revision': self.__revision__(report_text),
            'risk': self.__risk__(report_text),
            'report_text': report_text,
            'url': row['url'],
            'content_hash': hashlib.sha256(
                report_text.encode('utf-8')
            ).hexdigest(),
            'data_source': 'ths',
            'crawl_dt': crawl_dt,
            'extractor_version': self.extractor_version
        }

    def __get_data_from_ths_remote__(
        self,
        codes: Sequence[str],
        id_keys: Set[int],
        limit: Optional[int] = 50
    ) -> pd.DataFrame:
        """Fetches report IDs absent from the local table | 获取本地表中不存在的研报 ID"""
        source_codes = list(dict.fromkeys(
            self.__source_code__(code) for code in codes
        ))
        candidates: List[Dict[str, Any]] = []
        for source_code in source_codes:
            try:
                rows = self.__report_list__(source_code)
            except Exception as error:
                print(f'[quanta] report list failed <{source_code}>: {error}')
                continue
            rows = rows if limit is None else rows[:limit]
            for row in rows:
                try:
                    report_id = self.__report_id__(row.get('url', ''))
                except ValueError as error:
                    print(f'[quanta] invalid report skipped: {error}')
                    continue
                if report_id not in id_keys:
                    candidates.append(row)
                    id_keys.add(report_id)

        records: List[Dict[str, Any]] = []
        crawl_dt = pd.Timestamp.now().floor('s')
        for row in candidates:
            try:
                report_text = self.__report_text__(row['url'])
                records.append(self.__raw_record__(row, report_text, crawl_dt))
            except Exception as error:
                print(
                    f"[quanta] report body failed <{row.get('url', '')}>: "
                    f'{error}'
                )
        return pd.DataFrame.from_records(records)

    def pipeline(self, **kwargs: Any) -> pd.DataFrame:
        """Runs remote extraction and local standardization | 执行远端提取及本地标准化"""
        df = self.__get_data_from_ths_remote__(**kwargs)
        func = getattr(
            self,
            f'__data_standard_{self.table}__',
            self.__data_standard__
        )
        df = func(df, **kwargs)
        if df.empty:
            print(f'[quanta] warning: pipeline returned empty data for table <{self.table}>')
        return df
