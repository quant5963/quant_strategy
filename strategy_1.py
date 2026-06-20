# -*- coding: utf-8 -*-
"""
A股中小盘·TTM Squeeze 量价共振与主力锁仓选股雷达 (高积分资金流全功能版)
----------------------------------------------------------------------
核心重构：
1. 【高积分专用】：完美修复 Tushare 官方对 `moneyflow` 接口的严格参数校验限制。
2. 【筹码深度微积分】：完全继承博迁新材 5日主力资金微积分 (`L_Net_5d_sum`) 与散户退潮因子。
3. 【博迁洗盘精准狙击】：盘后冷酷识别“缩量回踩 + 主力锁仓”的二浪黄金坑。
"""

import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import talib
import tushare as ts
from tabulate import tabulate
from zzshare.client import DataApi


# 终端ANSI颜色定义
COLOR_RED = "\033[1;31m"
COLOR_GREEN = "\033[1;32m"
COLOR_YELLOW = "\033[1;33m"
COLOR_RESET = "\033[0m"

# 用户配置的高积分 Token
TUSHARE_TOKEN = ""
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
zz_api = DataApi()

class StrategyConfig:
    # 博迁专属缩量阈值（若设为0.55则严卡极端洗盘，若设为0.9则偏温和）
    SHRINK_RATIO = 0.55  # 原为0.9，现可自由切换
    
    # 均线锚点：1为MA20，2为MA30（切换剧本）
    MA_ANCHOR = 30  # 原为20，现可改为30匹配推演

def get_hot_score(ts_code, trade_date):
    score = 0.0
    try:
        # 2. zzshare同花顺热度（免费）
        ths_top = zz_api.ths_hot_top(date1=trade_date, top_n=100)
        if ts_code in ths_top['ts_code'].tolist():
            score += 0.8
    except:
        pass
    return min(score, 2.0)  # 封顶3分

def load_ths_hot_and_concept_matrix(trade_date):
    hot_dict = {}
    top_concepts = []
    stock_concept_map = {}

    try:
        print(f"📥 正在通过 DataApi 获取 {trade_date} 同花顺热度...")
        raw = zz_api.ths_hot_top(date1=trade_date, top_n=200)
        if not raw:
            print("⚠️ 返回空数据，尝试 AKShare 备用...")
            import akshare as ak
            raw = ak.stock_hot_rank_em()
            # 确保列名统一
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                raw.rename(columns={'股票代码': 'code', '排名': 'rank', '股票名称': 'name'}, inplace=True)
            else:
                raise ValueError("AKShare 也返回空")
        # 统一转为 DataFrame
        if isinstance(raw, pd.DataFrame):
            ths_top = raw
        else:
            ths_top = pd.DataFrame(raw)

        if ths_top.empty:
            return hot_dict, top_concepts, stock_concept_map

        # 打印列名（调试）
        print("📊 实际列名:", ths_top.columns.tolist())

        # 智能匹配（兼容多种列名）
        code_col = next((c for c in ['ts_code', 'code', 'stock_code', 'symbol', '股票代码'] if c in ths_top.columns), ths_top.columns[0])
        rank_col = next((c for c in ['rk', 'rank', 'order', '排名'] if c in ths_top.columns), None)

        # 标准化代码
        def std_code(c):
            c = str(c).strip().split('.')[0]
            if len(c) == 6:
                if c.startswith(('60', '68')):
                    return c + '.SH'
                elif c.startswith(('00', '30')):
                    return c + '.SZ'
            return c

        ths_top['std_code'] = ths_top[code_col].astype(str).apply(std_code)

        # 构建排名映射
        if rank_col:
            hot_dict = dict(zip(ths_top['std_code'], ths_top[rank_col].astype(int)))
        else:
            hot_dict = dict(zip(ths_top['std_code'], range(1, len(ths_top) + 1)))

        # 尝试获取概念（如果有）
        concept_col = next((c for c in ['concept', 'concepts', '概念', '所属概念'] if c in ths_top.columns), None)
        if concept_col:
            # 统计前5概念
            all_concepts = []
            for txt in ths_top[concept_col].dropna().astype(str):
                all_concepts.extend([c.strip() for c in txt.split(',') if c.strip() and c != 'nan'])
            if all_concepts:
                top_concepts = pd.Series(all_concepts).value_counts().head(5).index.tolist()
            # 个股映射
            for _, row in ths_top.iterrows():
                concepts = [c.strip() for c in str(row[concept_col]).split(',') if c.strip() and c != 'nan']
                if concepts:
                    stock_concept_map[row['std_code']] = concepts

        print(f"✅ 热点加载成功：{len(hot_dict)} 只个股排名，{len(top_concepts)} 个主线概念")

    except Exception as e:
        print(f"❌ 热点加载失败: {e}，本次略过热因子。")

    return hot_dict, top_concepts, stock_concept_map

def load_ths_hot_matrix(trade_date):
    """
    【核心优化】：在循环外一次性拉取同花顺热度榜，规避频次限制并建立映射字典
    """
    hot_dict = {}
    try:
        print(f"📥 正在通过 DataApi 获取 {trade_date} 同花顺全市场热度矩阵...")
        # 假设返回的 DataFrame 包含 'ts_code' 和 'rk' (排名) 字段
        ths_top = zz_api.ths_hot_top(date1=trade_date, top_n=200)
        if not ths_top.empty:
            # 如果接口返回的列名不是 rk，这里可以根据实际调整，通常按顺序即为排名
            if 'rk' not in ths_top.columns:
                ths_top['rk'] = range(1, len(ths_top) + 1)
            
            # 建立 股票代码 -> 排名的映射
            hot_dict = dict(zip(ths_top['ts_code'], ths_top['rk']))
            print(f"✅ 成功加载 {len(hot_dict)} 只热点个股数据。")
    except Exception as e:
        print(f"⚠️ DataApi 热度接口调用失败或超时: {e}，本次运行将略过热度因子。")
    return hot_dict

def get_last_trade_date():
    """获取最近一个交易日的日期"""
    return "20260617"
    today_str = datetime.now().strftime('%Y%m%d')
    search_back = 0
    while search_back < 15:
        check_date = (datetime.now() - timedelta(days=search_back)).strftime('%Y%m%d')
        df = pro.trade_cal(exchange='SSE', is_open=1, start_date=check_date, end_date=check_date)
        if not df.empty and df.iloc[0]['is_open'] == 1:
            return check_date
        search_back += 1
    return today_str

def run_post_market_pipeline(is_bull_market=True):
    print("=" * 80)
    print(f"🦅 启动《A股量价共振与主力锁仓选股雷达》 | 高积分全功能版")
    print("=" * 80)

    last_trade_date = get_last_trade_date()
    print(f"📅 当前分析交易日: {last_trade_date}")
    
    # 动态自适应阈值
    THRES_VR = 1.2 if is_bull_market else 1.5
    THRES_MA_VR = 1.2 if is_bull_market else 1.5
    THRES_SLOPE = 0.03 if is_bull_market else 0.05
    PASS_SCORE = 2.5 if is_bull_market else 3.0

    # 1. 在外部加载 DataApi 双层热点矩阵（个股热度 + 概念主线）
    hot_matrix, top_concepts, stock_concept_map = load_ths_hot_and_concept_matrix(last_trade_date)
    
    print("📥 正在加载全市场基础股票池(剔除ST、科创、北交、次新)...")
    try:
        df_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,list_date")
        df_basic = df_basic[~df_basic["name"].str.contains(r"ST|\*ST|退", na=False)]
        df_basic = df_basic[~df_basic["symbol"].str.startswith(("8", "43", "92"), na=False)]
        sixty_days_ago = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
        df_basic = df_basic[df_basic["list_date"] <= sixty_days_ago]
    except Exception as e:
        print(f"❌ 获取股票池失败: {e}")
        return

    print("📥 正在跨接口批量下载 60日 K线 + 资金流向大矩阵 (修复参数格式)...")
    try:
        start_history_dt = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        cal_df = pro.trade_cal(exchange='SSE', is_open=1, start_date=start_history_dt, end_date=last_trade_date)
        trade_days = cal_df['cal_date'].head(35).values 
        hist_mega_list = []
        for t_date in trade_days:
            # 严格转为规范的字符串格式，防止接口报错
            str_date = str(t_date).strip()
            
            # 接口 1: 市场基础日线
            d_sh = pro.daily(trade_date=str_date, exchange='SSE', fields="ts_code,trade_date,open,high,low,close,vol,pct_chg,amount")
            d_sz = pro.daily(trade_date=str_date, exchange='SZSE', fields="ts_code,trade_date,open,high,low,close,vol,pct_chg,amount")
            day_daily = pd.concat([d_sh, d_sz], ignore_index=True)
            
            # 接口 2: 基本指标（换手、市值）
            b_sh = pro.daily_basic(trade_date=str_date, exchange='SSE', fields="ts_code,turnover_rate,total_mv")
            b_sz = pro.daily_basic(trade_date=str_date, exchange='SZSE', fields="ts_code,turnover_rate,total_mv")
            day_basic = pd.concat([b_sh, b_sz], ignore_index=True)
            
            # 接口 3: 【方案 B 核心修复】显式传递规整后的 trade_date
            f_df = pro.moneyflow(trade_date=str_date, fields="ts_code,buy_lg_amount,buy_elg_amount,sell_lg_amount,sell_elg_amount,buy_sm_amount,sell_sm_amount")
            
            if not day_daily.empty and not day_basic.empty and not f_df.empty:
                m1 = pd.merge(day_daily, day_basic, on="ts_code")
                day_merged = pd.merge(m1, f_df, on="ts_code")
                hist_mega_list.append(day_merged)
            
            # 增加安全延时，高积分接口依然受单分钟调用频次约束
            time.sleep(0.25) 
            
        mega_df = pd.concat(hist_mega_list, ignore_index=True)
        mega_df = mega_df.sort_values(by=["ts_code", "trade_date"]).reset_index(drop=True)
    except Exception as e:
        print(f"❌ 矩阵多维数据拉取失败: {e}")
        return

    print("🧙‍♂️ 深度筹码融合矩阵就绪，特征微积分与决策树系统启动...")
    
    results = []
    grouped = mega_df.groupby("ts_code")
    
    name_dict = dict(zip(df_basic["ts_code"], df_basic["name"]))
    symbol_dict = dict(zip(df_basic["ts_code"], df_basic["symbol"]))

    for ts_code, sub_df in grouped:
        if ts_code not in name_dict or len(sub_df) < 30:
            continue
            
        close = sub_df["close"].values.astype(float)
        open_p = sub_df["open"].values.astype(float)
        high = sub_df["high"].values.astype(float)
        low = sub_df["low"].values.astype(float)
        volume = sub_df["vol"].values.astype(float)
        turnover = sub_df["turnover_rate"].values.astype(float)
        total_mv_yi = float(sub_df["total_mv"].iloc[-1]) / 10000.0 if pd.notna(sub_df["total_mv"].iloc[-1]) else 0.0
        pct_chg = float(sub_df["pct_chg"].iloc[-1]) if pd.notna(sub_df["pct_chg"].iloc[-1]) else 0.0
        amount = sub_df["amount"].values.astype(float)

        # 🔒 【拦截阀 1】：硬性风控拦截 —— 剔除放量大跌、大阴线
        if pct_chg < -2.0: continue  
        if close[-1] < open_p[-1]: continue  
        
        # 🔒 【拦截阀 2】：市值卡位 (50亿 - 1000亿)
        if total_mv_yi < 50.0 or total_mv_yi > 1000.0: continue

        # 🔒 【拦截阀 3】：涨停强行拦截 (涨停非低吸潜伏标的)
        if symbol_dict[ts_code].startswith(("30", "68")):
            if pct_chg >= 19.5: continue
        else:
            if pct_chg >= 9.5: continue

        # --- 筹码面：博迁策略大单/散户净量微积分计算 ---
        buy_lg = sub_df["buy_lg_amount"].values.astype(float)
        buy_elg = sub_df["buy_elg_amount"].values.astype(float)
        sell_lg = sub_df["sell_lg_amount"].values.astype(float)
        sell_elg = sub_df["sell_elg_amount"].values.astype(float)
        buy_sm = sub_df["buy_sm_amount"].values.astype(float)
        sell_sm = sub_df["sell_sm_amount"].values.astype(float)

        # 主力净流额与散户净流额
        main_net = (buy_lg + buy_elg) - (sell_lg + sell_elg)
        retail_net = buy_sm - sell_sm
        total_wan = amount / 10.0 # 千元转万元

        l_net_series = main_net / np.where(total_wan == 0, 1, total_wan)
        r_net_series = retail_net / np.where(total_wan == 0, 1, total_wan)

        # 5日滚动累计（彻底抹平主力日内对敲魔术）
        l_net_5d_sum = pd.Series(l_net_series).rolling(5).sum().values
        r_net_5d_sum = pd.Series(r_net_series).rolling(5).sum().values

        current_l_net_5d = l_net_5d_sum[-1]
        current_r_net_5d = r_net_5d_sum[-1]

        # 技术面多周期指标
        ma20 = talib.MA(close, timeperiod=20)
        ma30 = talib.MA(close, timeperiod=30)
        ma60 = talib.MA(close, timeperiod=60)
        v_ma20 = talib.MA(volume, timeperiod=20)
        
        # 🔒 【拦截阀 4】：多头趋势均线过滤
        if close[-1] < ma30[-1] and close[-1] < ma60[-1]: 
            continue  
        
        # 🔒 【拦截阀 5】：高换手熔断机制
        if turnover[-1] > 12.0:
            continue  
        
        # --- 量能特征：核心三维量能微积分计算 ---
        vr_series = []
        for i in range(5, len(volume)):
            avg_vol_5d = np.mean(volume[i-5:i])
            vr_series.append(volume[i] / max(avg_vol_5d, 1))
        vr_array = np.array([1.0]*5 + vr_series)

        ma_vr_array = talib.SMA(vr_array, timeperiod=5)
        vr_slope_array = talib.LINEARREG_SLOPE(ma_vr_array, timeperiod=3)

        current_vr = vr_array[-1]
        current_ma_vr = ma_vr_array[-1]
        current_slope = vr_slope_array[-1]

        # 动态量能一票否决
        if current_vr < THRES_VR and current_l_net_5d < 0.05:
            continue

        # 🧩 【框架灵魂乘法器公式还原】
        base_score = current_ma_vr * (1.0 + current_slope)

        # --- 补偿项：技术形态与筹码异动补偿得分 ---
        extra_score = 0.0
        tags = []
        
        # 【博迁筹码核心】：主力深度沉淀
        if current_l_net_5d > 0.08 and current_r_net_5d < 0:
            extra_score += 1.0
            tags.append("👑主力深沉淀")

        # TTM Squeeze 形态计算
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        kc_middle = talib.EMA(close, timeperiod=20)
        atr = talib.ATR(high, low, close, timeperiod=20)
        kc_upper = kc_middle + (1.5 * atr)
        kc_lower = kc_middle - (1.5 * atr)
        
        is_squeeze = (bb_upper[-1] < kc_upper[-1]) & (bb_lower[-1] > kc_lower[-1])
        was_squeeze_yesterday = (bb_upper[-2] < kc_upper[-2]) & (bb_lower[-2] > kc_lower[-2])

        if (not is_squeeze) and was_squeeze_yesterday and (close[-1] >= bb_upper[-1]):
            extra_score += 1.0  
            tags.append("💥Squeeze爆发")
        elif is_squeeze:
            extra_score += 0.5
            tags.append("🤫紧密蓄势")

        if close[-1] > ma20[-1]:
            extra_score += 0.5
            tags.append("强多头趋势")

        if current_slope > (THRES_SLOPE * 8):
            tags.append("⚡量能超速爆冲")

        # ---------------------------------------------------------------------
        # 🔥🔥🔥 【纵深重构：个股热度 + 概念风口双层赋分系统】
        # ---------------------------------------------------------------------
        hot_score = 0.0
        is_concept_hot = False
        
        # 维度一：个股所属概念是否处于“全市场5大主线风口”
        if ts_code in stock_concept_map:
            my_concepts = stock_concept_map[ts_code]
            hit_concepts = [c for c in my_concepts if c in top_concepts]
            if hit_concepts:
                is_concept_hot = True
                extra_score += 1.0  # 主线板块个股自带1分溢价（集团军光环）
                tags.append(f"风口主线({hit_concepts[0]})")

        # 维度二：个股自身的人气热度排名
        if ts_code in hot_matrix:
            rank = hot_matrix[ts_code]
            if rank <= 15:
                hot_score = 1.2
                tags.append(f"🔥市场总龙头(第{rank}名)")
            elif rank <= 50:
                hot_score = 0.8
                tags.append(f"🔥主线核心股(第{rank}名)")
            else:
                hot_score = 0.4
                tags.append(f"👀热点边缘(第{rank}名)")
        final_score = round(max(0.0, min(5.0, base_score + extra_score + hot_score)), 2)

        # 动态得分豁免：若有真金白银的主力锁仓，可略微降低技术形态及格线
        if final_score < PASS_SCORE and current_l_net_5d < 0.06:
            continue

        # =========================================================================
        # 🧠🧠🧠 【核心重构：智能化高级交易决策树 & Anti-FOMO 挂单狙击系统】 🧠🧠🧠
        # =========================================================================
        is_overheated = (pct_chg > 6.5) or (current_slope > 1.3)
        
        # 【博迁洗盘模型】：回踩均线区间 + 极度缩量 + 主力大单死守
        is_boqian_low_buy = (close[-1] <= ma30[-1] * 1.02) and \
                             (volume[-1] < v_ma20[-1] * 0.55) and \
                             (current_l_net_5d > 0.02)

        if is_boqian_low_buy:
            target_buy_price = round(close[-1], 2)
            decision = "🌟 黄金低吸！缩量锁仓"
            tags.insert(0, f"💎博迁洗盘模型(均线缩量), 明日开盘即吸纳({target_buy_price})")
            decision_color = COLOR_GREEN
        elif is_overheated:
            target_buy_price = round(ma20[-1], 2)
            decision = "🛑 拦截追高！限价低吸"
            tags.insert(0, f"🔥动能爆表, 锁定20日线挂单({target_buy_price})")
            decision_color = COLOR_RED
        elif current_l_net_5d > 0.05 and current_r_net_5d < 0:
            target_buy_price = round(low[-1], 2)
            decision = "🤫 主力锁仓！明日潜伏"
            tags.insert(0, f"📈主力暗中吸筹, 次日锁定今日最低价({target_buy_price})附近潜伏")
            decision_color = COLOR_YELLOW
        else:
            decision = "👀 适度跟踪"
            tags.insert(0, "横盘震荡, 大资金无显著倒灶动作")
            decision_color = COLOR_RESET

        chg_str = f"{pct_chg:+.2f}%"
        if pct_chg > 0: chg_str = f"{COLOR_RED}{chg_str}{COLOR_RESET}"

        score_str = f"{final_score:.2f}"
        if final_score >= 4.0:
            score_str = f"{COLOR_RED}{score_str}{COLOR_RESET}"
        elif final_score >= PASS_SCORE:
            score_str = f"{COLOR_YELLOW}{score_str}{COLOR_RESET}"
            
        decision_display = f"{decision_color}{decision}{COLOR_RESET}"

        results.append([
            symbol_dict[ts_code], name_dict[ts_code], f"{close[-1]:.2f}", chg_str,
            f"{total_mv_yi:.1f}亿", f"{turnover[-1]:.2f}%", score_str, decision_display, ",".join(tags) if tags else "—", final_score
        ])

    results.sort(key=lambda x: x[9], reverse=True)
    final_rows = [row[:-1] for row in results]
    headers = ["股票代码", "名称", "收盘价", "当日涨跌", "总市值", "换手率", "共振评分", "决策建议", "量价与筹码特征标签"]

    print("\n" + "=" * 32 + " 🏁 盘后雷达选股结算 " + "=" * 32)
    print(f"📊 实际并行清洗个股: {len(grouped)} 只 | 成功突围靠谱股票: {len(final_rows)} 只")
    
    lurking_rows = [row for row in results if "主力锁仓" in row[7]]  # row[7] 是带颜色的决策建议
    if lurking_rows:
        print("\n" + "=" * 32 + " 🤫 主力锁仓·明日潜伏精选 " + "=" * 32)
        # 去掉最后一列（原始评分），只显示前9列（与主表一致）
        lurking_display = [row[:-1] for row in lurking_rows]
        # 按评分降序（原results已排序，但为了保险可再排一次）
        # 但lurking_display没有评分，所以直接用原来的顺序
        lurking_display = [row[:-1] for row in lurking_rows[:30]]
        print(tabulate(lurking_display, headers=headers, tablefmt="grid"))
    else:
        print("\n⏳ 今日盘后未发现“主力锁仓·明日潜伏”类标的。")
    
    #if final_rows:
    #    print(tabulate(final_rows, headers=headers, tablefmt="grid"))
    #else:
    #    print(f"⏳ 今日盘后未发现完美符合多头量价共振或博迁锁仓模型的股票。")
    print("\n" + "-" * 80)

if __name__ == "__main__":
    run_post_market_pipeline(is_bull_market=True)