import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import optuna


def calculate_gain(prices, signals, n, initial_balance, to_print=False):
    prices = pd.Series(prices)
    signals = pd.Series(signals
                        )
    if prices.iloc[0] > initial_balance:
        initial_balance = max(
            initial_balance, prices.iloc[0])  # Starting capital
    balance = initial_balance
    position = 0               # Number of shares currently held
    buy_price = 0              # Price at which you buy (to compute gain)
    total_gain = 0             # Total profit/loss
    last_action = None         # Track the last action ('buy' or 'sell')

    for i in range(len(prices)):
        price = prices.iloc[i]   # Access price using iloc
        signal = signals.iloc[i]  # Access signal using iloc

        # Buy signal: signal > 0
        # Buy only if we don't already hold stock and didn't buy last
        if signal > n and position == 0 and last_action != 'buy':
            position = balance // price   # Buy as many shares as we can afford
            buy_price = price             # Record the price at which we buy
            # Deduct the cost of buying shares
            balance -= (position * price) + 3
            last_action = 'buy'           # Update last action to 'buy'

        # Sell signal: signal < 0
        # Sell only if we have shares to sell and didn't sell last
        elif signal < n and position > 0 and last_action != 'sell':
            # Sell all shares and update balance
            balance += (position * price) - 3
            # Compute the gain for this transaction
            total_gain += ((price - buy_price) * position) - 3
            position = 0                  # No more shares after selling
            last_action = 'sell'          # Update last action to 'sell'

    if initial_balance == 0:
        raise ValueError("Initial balance cannot be zero.")
    if prices.empty:
        raise ValueError("Prices series is empty.")

    # Calculate profit percentage
    profit_percentage = round(
        (balance + position * prices.iloc[-1] - initial_balance) / initial_balance * 100, 2)

    if to_print == True:
        print(
            f"Initial balance: {round(initial_balance, 1)}, Final balance: {
                round(balance, 1)}, "
            f"Profit: {profit_percentage}%, "
            f"N. of shares: {position}, Value in shares: {
                round(position * prices.iloc[-1], 1)}"
        )

    return profit_percentage


def stock_analysis(stock, buy_day=None, period=6):

    today_date = datetime.today()
    # today_date = today_date - relativedelta(months=1)
    start_date = today_date - relativedelta(months=period)
    df = yf.download(stock, start=start_date.strftime(
        '%Y-%m-%d'), end=today_date.strftime('%Y-%m-%d'))  # , interval="1d"

    # Function to calculate RSI

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def ichimoku_cloud(df):
        # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
        df['Tenkan_Sen'] = (df['High'][stock].rolling(
            window=9).max() + df['Low'][stock].rolling(window=9).min()) / 2

        # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
        df['Kijun_Sen'] = (df['High'][stock].rolling(
            window=26).max() + df['Low'][stock].rolling(window=26).min()) / 2

        # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
        df['Senkou_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)

        # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
        df['Senkou_B'] = ((df['High'][stock].rolling(window=52).max(
        ) + df['Low'][stock].rolling(window=52).min()) / 2).shift(26)

        # Lagging Span (Chikou Span): Close price shifted back by 26 periods
        df['Chikou_Span'] = df['Close'][stock].shift(-26)
        return df

    # Calculate indicators and signals
    close_values = df['Close'][stock]
    rsi = calculate_rsi(df['Close'][stock], period=14)
    rsi_buy_signal = (rsi < 30)
    rsi_sell_signal = (rsi > 70)

    ema_200 = df['Close'][stock].ewm(span=200, adjust=False).mean()
    ema_200_buy_signal = df['Close'][stock] > ema_200
    ema_200_sell_signal = df['Close'][stock] < ema_200

    ema_100 = df['Close'][stock].ewm(span=100, adjust=False).mean()
    ema_100_buy_signal = df['Close'][stock] > ema_100
    ema_100_sell_signal = df['Close'][stock] < ema_100

    ema_50 = df['Close'][stock].ewm(span=50, adjust=False).mean()
    ema_50_buy_signal = df['Close'][stock] > ema_50
    ema_50_sell_signal = df['Close'][stock] < ema_50

    ema_25 = df['Close'][stock].ewm(span=50, adjust=False).mean()
    ema_25_buy_signal = df['Close'][stock] > ema_25
    ema_25_sell_signal = df['Close'][stock] < ema_25

    ema_10 = df['Close'][stock].ewm(span=50, adjust=False).mean()
    ema_10_buy_signal = df['Close'][stock] > ema_10
    ema_10_sell_signal = df['Close'][stock] < ema_10

    mean_price = df['Close'][stock].rolling(window=20).mean()
    std_price = df['Close'][stock].rolling(window=20).std()
    z_score = (df['Close'][stock] - mean_price) / std_price
    # z_score_buy_signal = (z_score < -1)
    # z_score_sell_signal = (z_score > 1)

    ema_short = df['Close'][stock].ewm(span=12, adjust=False).mean()
    ema_long = df['Close'][stock].ewm(span=26, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_buy_signal = (macd_signal > 0)
    macd_sell_signal = (macd_signal < 0)

    momentum = df['Close'][stock].pct_change(periods=5)
    momentum_buy_signal = (momentum > 0)
    momentum_sell_signal = (momentum < 0)

    # Bollinger Bands calculation
    upper_band = mean_price + (std_price * 2)
    lower_band = mean_price - (std_price * 2)

    bollinger_buy_signal = close_values < lower_band
    bollinger_sell_signal = close_values > upper_band

    df = ichimoku_cloud(df)
    ichimoku_buy = (df['Close'][stock] > df['Senkou_A']) & (
        df['Close'][stock] > df['Senkou_B'])
    ichimoku_sell = (df['Close'][stock] < df['Senkou_A']) & (
        df['Close'][stock] < df['Senkou_B'])

    def transform_series(series):
        # Ensure there is at least one value in the series
        if series.empty:
            return series

        # Shift the series to compare with previous values
        prev = series.shift(1, fill_value=1)  # Fill the first value with 1
        return series.where((series * prev <= 0) & (prev != 0), 0)

    # Define the objective function for Optuna
    def objective(trial):
        # Suggest weights for each signal
        rsi_weight = trial.suggest_float("rsi_weight", -1, 1)
        trend_weight = trial.suggest_float("trend_weight", -1, 1)
        macd_weight = trial.suggest_float("macd_weight", -1, 1)
        # momentum_weight = trial.suggest_float("momentum_weight", -1, 1)
        bollinger_weight = trial.suggest_float("bollinger_weight", -1, 1)
        ichimoku_weight = trial.suggest_float("ichimoku_weight", -1, 1)

        # Normalize weights to ensure they sum to 1
        weights = [rsi_weight, trend_weight, macd_weight,
                   bollinger_weight, ichimoku_weight]
        #    momentum_weight,
        total_weight = sum(abs(w) for w in weights)

        if total_weight == 0:
            print("Total weight is zero!")
            return float('inf')

        normalized_weights = [w / total_weight for w in weights]

        # Unpack normalized weights
        rsi_weight, trend_weight, macd_weight, bollinger_weight, ichimoku_weight = normalized_weights  # momentum_weight

        # Calculate trend signals
        trend_buy = (
            trend_weight * (
                ema_200_buy_signal.astype(int) +
                ema_100_buy_signal.astype(int) +
                ema_50_buy_signal.astype(int) +
                ema_25_buy_signal.astype(int)
                # ema_10_buy_signal.astype(int)
            )
        )
        trend_sell = (
            trend_weight * (
                ema_200_sell_signal.astype(int) +
                ema_100_sell_signal.astype(int) +
                ema_50_sell_signal.astype(int) +
                ema_25_sell_signal.astype(int)
                # ema_10_sell_signal.astype(int)
            )
        )

        # Calculate buy and sell counts
        buy_count = (
            rsi_weight * rsi_buy_signal.astype(int) +
            trend_buy +
            macd_weight * macd_buy_signal.astype(int) +
            # momentum_weight * momentum_buy_signal.astype(int) +
            bollinger_weight * bollinger_buy_signal.astype(int) +
            ichimoku_weight * ichimoku_buy.astype(int)
        )
        sell_count = (
            rsi_weight * rsi_sell_signal.astype(int) +
            trend_sell +
            macd_weight * macd_sell_signal.astype(int) +
            # momentum_weight * momentum_sell_signal.astype(int) +
            bollinger_weight * bollinger_sell_signal.astype(int) +
            ichimoku_weight * ichimoku_sell.astype(int)
        )

        # Calculate net signal
        net_signal = buy_count - sell_count
        net_signal = transform_series(net_signal)
        if buy_day:
            net_signal[f'{buy_day}+00:00'] = max(net_signal)
        # net_signal.iloc[0] = max(net_signal)

        # Calculate gain using the net signal
        gain = calculate_gain(close_values, net_signal,
                              n=0, initial_balance=300)
        return -gain  # Negate for maximization

    # Create the Optuna study
    # Minimize the negative gain
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)  # Run for 100 trials

    # Extract the best parameters
    optimal_weights = study.best_params
    max_profit = -study.best_value  # Undo negation to get the actual maximum profit
    # print("Optimal Weights:", optimal_weights)
    print("Maximum Profit (%):", max_profit)

    rsi_weight, trend_weight, macd_weight, bollinger_weight, ichimoku_weight = optimal_weights.values()
    # momentum_weight,

    # Calculate trend signals
    trend_buy = (
        trend_weight * (
            ema_200_buy_signal.astype(int) +
            ema_100_buy_signal.astype(int) +
            ema_50_buy_signal.astype(int) +
            ema_25_buy_signal.astype(int)
            # ema_10_buy_signal.astype(int)
        )
    )
    trend_sell = (
        trend_weight * (
            ema_200_sell_signal.astype(int) +
            ema_100_sell_signal.astype(int) +
            ema_50_sell_signal.astype(int) +
            ema_25_sell_signal.astype(int)
            # ema_10_sell_signal.astype(int)
        )
    )

    # Calculate buy and sell counts
    buy_count = (
        rsi_weight * rsi_buy_signal.astype(int) +
        trend_buy +
        macd_weight * macd_buy_signal.astype(int) +
        # momentum_weight * momentum_buy_signal.astype(int) +
        bollinger_weight * bollinger_buy_signal.astype(int) +
        ichimoku_weight * ichimoku_buy.astype(int)
    )
    sell_count = (
        rsi_weight * rsi_sell_signal.astype(int) +
        trend_sell +
        macd_weight * macd_sell_signal.astype(int) +
        # momentum_weight * momentum_sell_signal.astype(int) +
        bollinger_weight * bollinger_sell_signal.astype(int) +
        ichimoku_weight * ichimoku_sell.astype(int)
    )

    def shift_series(series):
        new_element = pd.Series([1])
        shifted_series = pd.concat(
            [new_element, series[:-1]], ignore_index=True)
        return pd.Series(shifted_series, index=df.index)

    net_signal = buy_count - sell_count
    net_signal = transform_series(net_signal)
    # net_signal = shift_series(net_signal)
    if buy_day:
        net_signal[f'{buy_day}+00:00'] = max(net_signal)
    # net_signal.iloc[0] = max(net_signal)

    n = 0
    calculate_gain(close_values, net_signal, n, 300, True)

    # Create EMA 200 trace
    ema_200_trace = go.Scatter(
        x=df.index, y=ema_200, mode='lines', name='EMA 200', line=dict(color='orangered'))
    ema_100_trace = go.Scatter(
        x=df.index, y=ema_100, mode='lines', name='EMA 100', line=dict(color='darkorange'))
    ema_50_trace = go.Scatter(
        x=df.index, y=ema_50, mode='lines', name='EMA 50', line=dict(color='orange'))
    ema_25_trace = go.Scatter(
        x=df.index, y=ema_25, mode='lines', name='EMA 25', line=dict(color='gold'))

    # Create the plot
    fig = go.Figure()

    # Add the close price line
    fig.add_trace(go.Scatter(x=df.index, y=close_values, mode='lines+markers',
                             name='Close Price', line=dict(color='blue')))
    fig.add_trace(ema_200_trace)
    # fig.add_trace(ema_100_trace)
    # fig.add_trace(ema_50_trace)
    fig.add_trace(ema_25_trace)

    # for i in range(len(vp)):
    #     fig.add_trace(go.Bar(x=[vp['Volume'][i]],
    #                          y=[vp['Price_Level'][i]],
    #                          orientation='h',
    #                          marker=dict(color='rgba(100, 100, 255, 0.6)'),
    #                          showlegend=False))

    # Normalize net_signal for colorscale
    max_signal = max(np.abs(net_signal.max()), np.abs(net_signal.min()))
    normalized_signal = np.abs(net_signal) / max_signal

    # Define the colorscales
    colorscaleG = [[
        0, "lightgreen"], [0.5, "green"], [1, "darkgreen"]]
    colorscaleR = [[
        0, "lightcoral"], [0.5, "red"], [1, "darkred"]]

    # Add buy markers
    fig.add_trace(go.Scatter(
        x=df.index[net_signal > n],
        y=close_values[net_signal > n],
        mode='markers',
        name='Buy Suggestion',
        marker=dict(
            color=normalized_signal[net_signal > n],
            colorscale=colorscaleG,
            symbol='triangle-up',
            size=10
        )
    ))

    # Add sell markers
    fig.add_trace(go.Scatter(
        x=df.index[net_signal < -n],
        y=close_values[net_signal < -n],
        mode='markers',
        name='Sell Suggestion',
        marker=dict(
            color=normalized_signal[net_signal < -n],
            colorscale=colorscaleR,
            symbol='triangle-down',
            size=10
        )
    ))

    if buy_day:
        fig.add_vline(
            x=buy_day,
            line=dict(color="red", width=2, dash="dot"),  # Custom line style
        )

    # Update layout
    fig.update_layout(
        title=stock,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=400,
        width=1100
    )

    # Show the plot
    fig.show()

    return fig
