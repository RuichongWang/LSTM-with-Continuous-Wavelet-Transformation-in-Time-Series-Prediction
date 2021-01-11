import os
import threading
from itertools import chain
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

file_path = 'YOUR_PATH'
os.chdir(file_path)

# get all stock codes
# http://www.sse.com.cn/assortment/stock/list/share/
# http://www.szse.cn/market/product/stock/list/index.html

sz_code = pd.read_excel("SZA.xlsx", usecols=['公司代码', 'A股上市日期'])
sh_code = pd.read_csv("SHA.csv", names=['上市日期']).reset_index().iloc[:, :2]
sh_code.columns = ['公司代码', 'A股上市日期']

sh_code['flag'] = '0'
sz_code['flag'] = '1'  # the url to retrive the data is different in two market (SZA & SHA)
code_all = pd.concat((sz_code, sh_code)).values


def get_data(code_list, k):
    for i, (code, _, flag) in enumerate(code_list):
        code = str(code).zfill(6)
        if flag == 1:
            url = 'ADDRESS_HIDDEN?code=1%s&start=20170630end=20201231&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP' % (
                flag + code)
        else:
            url = 'ADDRESS_HIDDEN?code=0%s&start=20170630end=20201231&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP' % (
                flag + code)
        filename = '%s.csv' % code
        urlretrieve(url, filename)


def parallel_run(func, file_list, threads):
    steps = int(len(file_list) / (threads)) + 1
    num = int(len(file_list) / steps) + 1
    for k in range(num):
        name = 't' + str(k)
        locals()['t' + str(k)] = threading.Thread(
            target=func,
            args=(file_list[steps * k:min(steps * (k + 1), len(file_list))],
                  k))
    for k in range(num):
        locals()['t' + str(k)].start()
    for k in range(num):
        locals()['t' + str(k)].join()


parallel_run(get_data, code_all, 50)  # download data

# merging
files = os.listdir()
files = [x for x in files if 'S' not in x]  # skip SHA.csv and SZA.csv
train = []
test  = []

for i, file in enumerate(files):
    with open(file, encoding='GB2312') as f:
        data = f.read()
        data = data.split('\n')[1:-1]
        if len(data) > 250:
            data = data[:
                        -125][::
                              -1]  # skip the first 6 months, ~125 trading days
            data = list(map(lambda x: np.array(x.split(',')), data))
            data = list(filter(lambda x: int(x[0].split('-')[0]) > 2016, data))
            data = list(
                map(lambda x: x[[0, 1, 3, 9, 11, 13]], data)
            )  # keep date, code, close price, daily return, volume and total cap

            train.extend(data[:-250])  # data of in the last year is test data
            test.extend(data[-250:])

columns = [
    'date', 'code', 'close_price', 'daily_return', 'trade_volume',
    'liquid_market_value'
]
train = pd.DataFrame(train, columns=columns)
test  = pd.DataFrame(test, columns=columns)


def data_cleaning(df):
    df['code'] = df.code.map(lambda x: x.replace("'", "")).astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df.daily_return != 'None']
    df['daily_return'] = df['daily_return'].astype(float)
    df.sort_values(['code', 'date'], inplace=True)
    return df


train = data_cleaning(train)
test  = data_cleaning(test)


def single_stock_sample_generator(df):
    """
    for a 33-days-data-block
    we first check if there is any abnormal things exsits
    then we generate x and y for this block
    """
    liquid_check = df.trade_volume.min(
    ) == 0  # skip stocks that are not trading during the selected days
    market_cap_check = df.liquid_market_value.min(
    ) < 9.75 * 10**8  # skip stocks whose liquid_market_value is too small, those stocks have very high volatility and will influence data distribution
    price_check = df.close_price.min(
    ) < 10  # skip stocks whose price is too low, since the daily return is our target and the minimum price change is 0.01,
    # stock's return rate will be influenced a lot by a 0.01 change for low price stock
    if liquid_check or market_cap_check or price_check:  # not qualified data will be skipped
        return np.array((None, None))

    x_temp = df.iloc[:30]['daily_return'].values
    yesterday_price = df.iloc[29]['close_price'].values
    target_price = df.iloc[-1]['close_price'].values
    y_temp = (target_price - yesterday_price) / yesterday_price * 10
    return np.array((x_temp, y_temp))


def all_stock_generator(df):
    """
    for a single stock
    we seperate every 33 days data (30 history data + 3 forcast data)
    into a new dataframe
    """
    df['group_no'] = list(range(len(df)))
    df['group_no'] = df['group_no'] // 33
    df = df[df.group_no <
            max(df.group_no
                )]  # skip the last group which may be less than 33 days data
    res = df.groupby('group_no').apply(single_stock_sample_generator).values
    res = list(chain.from_iterable(np.array(
        (res))))  # transform a lists of list to just a list
    return res


train_res = train.groupby('code').apply(
    all_stock_generator
)  # apply it to all stocks, the result is a list of 3600 stocks' list which contains the x and y of each stock
del train  # save some ram

test_res = test.groupby('code').apply(
    all_stock_generator
)  # apply it to all stocks, the result is a list of 3600 stocks' list which contains the x and y of each stock
del test

np.save('train', train_res)
np.save('test', test_res)
