import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# 数据获取与预处理
def get_data():
    # 模拟数据获取，这里可以替换为实际数据获取代码，如读取CSV文件或API获取数据
    dates = pd.date_range(start='2020-01-01', periods=1000)
    prices = np.random.normal(100, 10, len(dates))
    volumes = np.random.normal(1000, 200, len(dates))
    data = pd.DataFrame({'Date': dates, 'Price': prices, 'Volume': volumes})
    data.set_index('Date', inplace=True)
    return data

data = get_data()

# 遗传算法优化交易策略参数
def fitness_function(params, data):
    short_window, long_window = params
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Price']
    signals['short_mavg'] = data['Price'].rolling(window=int(short_window)).mean()
    signals['long_mavg'] = data['Price'].rolling(window=int(long_window)).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    initial_capital = 100000.0
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Price'] = signals['signal']
    portfolio = positions.multiply(signals['price'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(signals['price'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['price'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    return portfolio['total'].iloc[-1]

def genetic_algorithm(population_size, generations, mutation_rate, data):
    population = [np.random.randint(1, 100, 2) for _ in range(population_size)]

    for gen in range(generations):
        fitness_scores = [fitness_function(ind, data) for ind in population]
        population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        next_gen = population[:population_size//2]

        for i in range(population_size//2):
            parent1, parent2 = random.sample(next_gen, 2)
            cross_point = random.randint(0, len(parent1)-1)
            child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
            child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
            if random.random() < mutation_rate:
                child1[random.randint(0, len(child1)-1)] = random.randint(1, 100)
            if random.random() < mutation_rate:
                child2[random.randint(0, len(child2)-1)] = random.randint(1, 100)
            next_gen.extend([child1, child2])

        population = next_gen

    best_solution = population[0]
    best_fitness = fitness_function(best_solution, data)
    return best_solution, best_fitness

best_params, best_fitness = genetic_algorithm(100, 50, 0.1, data)
print(f"Best parameters: {best_params}, Fitness: {best_fitness}")

# 模糊逻辑识别市场状态
def fuzzy_market_state(price, volume):
    price_antecedent = ctrl.Antecedent(np.arange(0, 100, 1), 'price')
    volume_antecedent = ctrl.Antecedent(np.arange(0, 10000, 100), 'volume')
    market_state = ctrl.Consequent(np.arange(0, 100, 1), 'market_state')

    price_antecedent['low'] = fuzz.trapmf(price_antecedent.universe, [0, 0, 20, 40])
    price_antecedent['medium'] = fuzz.trapmf(price_antecedent.universe, [20, 40, 60, 80])
    price_antecedent['high'] = fuzz.trapmf(price_antecedent.universe, [60, 80, 100, 100])

    volume_antecedent['low'] = fuzz.trapmf(volume_antecedent.universe, [0, 0, 2000, 4000])
    volume_antecedent['medium'] = fuzz.trapmf(volume_antecedent.universe, [2000, 4000, 6000, 8000])
    volume_antecedent['high'] = fuzz.trapmf(volume_antecedent.universe, [6000, 8000, 10000, 10000])

    market_state['bearish'] = fuzz.trapmf(market_state.universe, [0, 0, 20, 40])
    market_state['neutral'] = fuzz.trapmf(market_state.universe, [20, 40, 60, 80])
    market_state['bullish'] = fuzz.trapmf(market_state.universe, [60, 80, 100, 100])

    rule1 = ctrl.Rule(price_antecedent['low'] & volume_antecedent['low'], market_state['bearish'])
    rule2 = ctrl.Rule(price_antecedent['medium'] & volume_antecedent['medium'], market_state['neutral'])
    rule3 = ctrl.Rule(price_antecedent['high'] & volume_antecedent['high'], market_state['bullish'])

    market_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    market_simulation = ctrl.ControlSystemSimulation(market_ctrl)

    market_simulation.input['price'] = price
    market_simulation.input['volume'] = volume

    market_simulation.compute()
    return market_simulation.output['market_state']

# 神经网络生成交易信号
def build_neural_network():
    model = Sequential([
        Dense(64, input_dim=2, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32)

def generate_signal(model, price, volume):
    input_data = np.array([[price, volume]])
    prediction = model.predict(input_data)
    signal = np.argmax(prediction)
    return signal

# 生成训练数据
data['MarketState'] = data.apply(lambda row: fuzzy_market_state(row['Price'], row['Volume']), axis=1)
X_train = data[['Price', 'Volume']].values
y_train = data['MarketState'].values

# 构建和训练神经网络
model = build_neural_network()
train_neural_network(model, X_train, y_train)

# 生成交易信号并应用风险管理
data['Signal'] = data.apply(lambda row: generate_signal(model, row['Price'], row['Volume']), axis=1)
initial_capital = 100000.0
positions = pd.DataFrame(index=data.index).fillna(0.0)
positions['Price'] = data['Signal'].apply(lambda x: 1 if x == 2 else -1 if x == 0 else 0)
portfolio = positions.multiply(data['Price'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(data['Price'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Price'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

# 评估系统性能
total_return = portfolio['total'].iloc[-1]
max_drawdown = portfolio['total'].min()
win_rate = (data['Signal'] == 2).sum() / len(data)
print(f"Total return: {total_return}, Max drawdown: {max_drawdown}, Win rate: {win_rate}")

# 可视化结果
plt.figure(figsize=(10, 5))
plt.plot(portfolio['total'], label='Portfolio Total Value')
plt.title('Portfolio Performance')
plt.xlabel('Date')
plt.ylabel('Total Value')
plt.legend()
plt.show()

# 可视化市场状态和交易信号
plt.figure(figsize=(15, 7))

plt.subplot(3, 1, 1)
plt.plot(data.index, data['Price'], label='Price')
plt.title('Price Over Time')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data.index, data['MarketState'], label='Market State', color='orange')
plt.title('Market State Over Time')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data.index, data['Signal'], label='Signal', color='green')
plt.title('Trading Signals Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# 进一步性能评估
def evaluate_performance(portfolio):
    returns = portfolio['returns']
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility

    max_drawdown = 0
    peak = portfolio['total'][0]
    for value in portfolio['total']:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return annual_return, annual_volatility, sharpe_ratio, max_drawdown

annual_return, annual_volatility, sharpe_ratio, max_drawdown = evaluate_performance(portfolio)
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# 保存模型
model.save('trading_model.h5')