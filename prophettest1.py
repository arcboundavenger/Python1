import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
import utils
import datetime as dt
import itertools
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

df = pd.read_csv('test1securitybreach.csv')  # read data
# df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
df['floor'] = 1
df['cap'] = 47163

est_impact = int(input('Enter the No. of impact (1. Low; 2. Mid; 3. High; 4. Ultra High):'))

# create m and fit data
# promotion = df['promotion']
# data_with_regressors = add_regressor(df, promotion, varname='promotion')

announcement_date = input('Enter the announcement date (YYYYMMDD):')

announcement = pd.DataFrame({
    'holiday': 'announcement',
    'ds': pd.to_datetime([announcement_date]),
    'lower_window': -1 * est_impact,
    'upper_window': est_impact * 4, 'holidays_prior_scale': 5,
})

# expo = pd.DataFrame({
#   'holiday': 'expo',
#   'ds': pd.to_datetime(['20221004','20221206','20230111']),
#   'lower_window': -3,
#   'upper_window': 7, 'holidays_prior_scale': 2,
# })

# OBT = pd.DataFrame({
#     'holiday': 'OBT',
#     'ds': pd.to_datetime(['20210807']),
#     'lower_window': -10,
#     'upper_window': 30, 'holidays_prior_scale': 10,
# })

release_date = input('Enter the release date (YYYYMMDD):')

release = pd.DataFrame({
    'holiday': 'release',
    'ds': pd.to_datetime([release_date]),
    'lower_window': -5 * est_impact,
    'upper_window': est_impact * 5 * 3, 'holidays_prior_scale': 10,
})

holidays = pd.concat((announcement,
                      # expo,
                      # OBT,
                      release
                      ))

param_grid = {
    'growth': ['logistic'],
    'seasonality_prior_scale': [0.01,0.1,1],
    'interval_width': [0.8],
    'seasonality_mode': ['additive'],
    'changepoint_range': [i / 10 for i in range(3, 10, 1)],
    'changepoint_prior_scale': [0.01,0.05,0.1]
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
# maes = []  # Store the MAE for each params here
mapes = []  # Store the mape for each params here

for params in all_params:
    m = Prophet(**params).fit(df)
    df_cv = cross_validation(
        m,
        initial='90 days',
        period='15 days',
        horizon='30 days',
        parallel='threads'
    )
    df_p = performance_metrics(df_cv,
                               rolling_window=0.1
                               )
    # maes.append(df_p['mae'].values[0])
    mapes.append(df_p['mape'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
# tuning_results['mae'] = maes
tuning_results['mape'] = mapes
print(tuning_results.to_string())

best_params = all_params[np.argmin(mapes)]
print(best_params)

m = Prophet(
    daily_seasonality=False,
    growth=best_params['growth'],
    holidays=holidays,
    weekly_seasonality=True,
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    seasonality_mode=best_params['seasonality_mode'],
    interval_width=best_params['interval_width'],
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    changepoint_range=best_params['changepoint_range']
)

# m = Prophet(
#     daily_seasonality=False,
#     growth='logistic',
#     holidays=holidays,
#     weekly_seasonality=True,
#     seasonality_prior_scale=0.01,
#     seasonality_mode='additive',
#     interval_width=0.8,
#     changepoint_range=0.8,
#     changepoint_prior_scale=0.1
# )

# m.add_regressor('promotion', prior_scale= 0.01)
m.fit(df)

future = m.make_future_dataframe(periods=90)

# df_temp=pd.read_csv('promotiontest1.csv')
# future['promotion'] = df_temp['promotion']
future = future.fillna(0)
future['floor'] = 1
future['cap'] = 47163

forecast = m.predict(future)

from prophet.plot import plot_cross_validation_metric

# fig = plot_cross_validation_metric(df_cv, metric='mape', rolling_window=0.1)
# fig.show()

# fig = px.line(df, x="ds", y="y")
# fig.show()
fig1 = m.plot(forecast)
# fig2 = m.plot_components(forecast)

# plt.axvline(dt.datetime(2022, 8, 28), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 9, 7), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 9, 30), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 10, 10), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 10, 24), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 11, 2), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 11, 21), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 11, 30), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2022, 12, 21), ls='--', lw=1, c='red')
# plt.axvline(dt.datetime(2023, 1, 6), ls='--', lw=1, c='red')
plt.ylim(-50, max(df.y) * 1.1)
plt.show()
forecast.to_csv('forecast111.csv')
