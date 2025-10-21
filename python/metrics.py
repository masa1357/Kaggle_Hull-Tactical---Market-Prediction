import numpy as np
import pandas as pd
from typing import Union

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, intermediate_res:list = None) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Returns:
        float: The calculated adjusted Sharpe ratio.
    """
    solution = solution.copy().reset_index(drop=True)
    submission = submission.copy().reset_index(drop=True)
    solution['position'] = submission['prediction']

    # ありえない値を除外する (0 <= position <= 2)
        # 0 means that we don't invest in S & P at all but get only the risk-free rate.
        # 1 means that we invest all our money in S & P.
        # 2 means that we invest twice our capital in S & P while taking a credit at the risk-free rate.
        # -> つまり，普通に預金するか，S&Pに投資するか，S&Pに2倍レバレッジで投資するか（借金）の割合
    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    # Calculate strategy returns
    # フェデラルファンド金利(利息) * (1-予測値) + 予測値 * S&P500の翌日のリターン = 戦略のリターン(割合)
    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    # リターンとその標準偏差を用いてシャープレシオ（リスクあたりの効率）を計算
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate'] # 超過リターン -> 今回の戦略で得た割合から，リスクフリー時の割合を引いた分
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod() # 累積超過リターン -> 全期間の超過リターンをかけ合わせた分(1+で倍率に変換)
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1 # 平均超過リターン -> 複利は幾何平均で求める． また，倍率から割合に戻してる
    strategy_std = solution['strategy_returns'].std() # リターンの標準偏差

    trading_days_per_yr = 252 # 1年あたりの取引日数(固定値)
    if strategy_std == 0:
        raise ZeroDivisionError
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr) # 年率換算したシャープレシオ. sqrt(252)をかけることで年率換算している（統計的な性質らしい）
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)  # 年率換算したボラティリティ(価格変動率)

    # Calculate market return and volatility
    # S&P500に投資し続けた場合のリターンとボラティリティを計算
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate'] # S&P500が利息を上回る割合
    market_excess_cumulative = (1 + market_excess_returns).prod() # ↑の累積
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1 # train: 0.0003066067595838273 幾何平均，割合化
    market_std = solution['forward_returns'].std() # S&P500のリターンの標準偏差
    
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100) # train: 16.748459963166347 %
    
    # Calculate the volatility penalty
    # ボラティリティペナルティを計算
    # -> 市場のボラティリティの1.2倍を超える場合のペナルティ
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    # リターンペナルティを計算
    # -> 市場のリターンを下回る場合のペナルティ
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    # ペナルティ値の反映
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)

    # print("strategy_excess_returns NaN数:", solution['strategy_returns'].isna().sum())
    # print("strategy_std:", strategy_std)
    # print("strategy_excess_cumulative:", strategy_excess_cumulative)
    # print("market_excess_cumulative:", market_excess_cumulative)
    # print("adjusted_sharpe:", adjusted_sharpe)
    try:
        intermediate_res.append((strategy_mean_excess_return, strategy_std, sharpe, vol_penalty, return_penalty)) # 各値を記録(debug)
        return min(float(adjusted_sharpe), 1_000_000), intermediate_res # float変換，上限100万
    except NameError:
        return min(float(adjusted_sharpe), 1_000_000) # float変換，上限100万