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
from prophet.plot import add_changepoints_to_plot
import jenkspy

df2 = pd.read_csv('test1ff7core.csv')
df = df2  # read data

df2['ds'] = pd.to_datetime(df2['ds'])
df2.set_index(df2['ds'], inplace = True)
ts = df2['y']
y = np.array(ts.tolist())
n_breaks = int(input('Enter the number of changepoints:'))
breaks = jenkspy.jenks_breaks(y, n_breaks-1)
breaks_jkp = []
breaks_jkp_str = []
for v in breaks:
    idx = ts.index[ts == v]
    breaks_jkp.append(idx)
    breaks_jkp_str.append(idx.strftime('%Y-%m-%d')[0]) #Auto-detect changepoints

# df2['y'] = (df2['y'] - df2['y'].min()) / (df2['y'].max() - df2['y'].min())
df['floor'] = 1
df['cap'] = max(df.y) * 1.2

est_impact = int(input('Enter the No. of impact (1. Super Low; 2. Low; 3. Mid; 4. High; 5. Super High):'))

# create m and fit data
# promotion = df2['promotion']

announcement_date = input('Enter the announcement date (YYYYMMDD):')

announcement = pd.DataFrame({
    'holiday': 'announcement',
    'ds': pd.to_datetime([announcement_date]),
    'lower_window': -1 * est_impact,
    'upper_window': est_impact * 4, 'holidays_prior_scale': df.y.iloc[0] / max(df.y) * 10,
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
    'seasonality_prior_scale': [0.01, 0.1],
    'changepoint_range': [i / 10 for i in range(7, 10, 1)],
    'changepoint_prior_scale': [0.05, 0.1, 0.5, 1]
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
    growth='linear',
    holidays=holidays,
    weekly_seasonality=True,
    yearly_seasonality=False,
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    seasonality_mode='additive',
    interval_width=0.8,
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    changepoint_range=best_params['changepoint_range'],
    changepoints=breaks_jkp_str
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

# m.add_regressor('promotion', prior_scale=best_params['changepoint_prior_scale'])
m.fit(df)

future = m.make_future_dataframe(periods=30)

# df_temp = pd.read_csv('promotiontest1ff7core.csv')
# future['promotion'] = df_temp['promotion']
future = future.fillna(0)
future['floor'] = 1
future['cap'] = max(df.y) * 1.2

forecast = m.predict(future)

from prophet.plot import plot_cross_validation_metric

# fig = plot_cross_validation_metric(df_cv, metric='mape', rolling_window=0.1)
# fig.show()

fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast)

# plt.axvline(dt.datetime(2022, 8, 28), ls='--', lw=1, c='red')

plt.ylim(-50, max(df.y) * 1.2)
plt.show()
# fig = m.plot_components(forecast)
# fig.show()
forecast.to_csv('forecast111.csv')
