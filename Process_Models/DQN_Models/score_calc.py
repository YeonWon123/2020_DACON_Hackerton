import pandas as pd
import numpy as np


class Calculator:
    def __init__(self, submission):
        self.submission = submission
        self.sample_submission = pd.read_csv('module/sample_submission.csv')
        self.max_count = pd.read_csv('module/max_count.csv')
        self.stock = pd.read_csv('module/stock.csv')
        order = pd.read_csv('module/order.csv', index_col=0)
        order.index = pd.to_datetime(order.index)
        self.order = order

    def get_state(self, data):
        if 'CHECK' in data:
            return int(data[-1])
        elif 'CHANGE' in data:
            return int(data[-1])
        else:
            return np.nan
    
    @staticmethod
    def F(x, a):
        if x < a:
            return 1 - x/a
        else:
            return 0

    def subprocess(self, df):
        """숫자 인덱스를 제거하고 타임을 인덱스로 사용한다"""
        out = df.copy()
        column = 'time'

        out.index = pd.to_datetime(out[column])
        out = out.drop([column], axis=1)
        out.index.name = column
        return out

    def cal_schedule_part_1(self, df):
        columns = ['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']
        df_set = df[columns]
        df_out = df_set * 0
        
        p = 0.985
        dt = pd.Timedelta(days=23)
        end_time = df_out.index[-1]

        for time in df_out.index:
            out_time = time + dt
            if end_time < out_time:
                break
            else:            
                for column in columns:
                    set_num = df_set.loc[time, column]
                    if set_num > 0:
                        out_num = np.sum(np.random.choice(2, set_num, p=[1-p, p]))         
                        df_out.loc[out_time, column] = out_num

        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0
        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out

    def cal_schedule_part_2(self, df, line='A'):
        if line == 'A':
            columns = ['Event_A', 'MOL_A']
        elif line == 'B':
            columns = ['Event_B', 'MOL_B']
        else:
            columns = ['Event_A', 'MOL_A']
            
        schedule = df[columns].copy()
        
        schedule['state'] = 0
        schedule['state'] = schedule[columns[0]].apply(lambda x: self.get_state(x))
        schedule['state'] = schedule['state'].fillna(method='ffill')
        schedule['state'] = schedule['state'].fillna(0)
        
        schedule_process = schedule.loc[schedule[columns[0]]=='PROCESS']
        df_out = schedule.drop(schedule.columns, axis=1)
        df_out['PRT_1'] = 0.0
        df_out['PRT_2'] = 0.0
        df_out['PRT_3'] = 0.0
        df_out['PRT_4'] = 0.0
        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0

        p = 0.975
        times = schedule_process.index

        c = 0
        c_n = 0
        s = 0
        s_n = 0
        for i, time in enumerate(times):
            if 'CHANGE' in schedule[columns[0]][i]:
                c += 1
                if i == 0 or schedule[columns[0]][i] != schedule[columns[0]][i- 1]:
                    c_n += 1
            elif 'STOP' in schedule[columns[0]][i]:
                s += 1
                if i == 0 or schedule[columns[0]][i] != schedule[columns[0]][i- 1]:
                    s_n += 1
            value = schedule.loc[time, columns[1]]
            state = int(schedule.loc[time, 'state'])
            df_out.loc[time, 'PRT_'+str(state)] = -value
            if i+48 < len(times):
                out_time = times[i+48]
                df_out.loc[out_time, 'MOL_'+str(state)] = value*p

        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out

    def cal_stock(self, df, df_order):
        df_stock = df * 0

        blk2mol = {}
        blk2mol['BLK_1'] = 'MOL_1'
        blk2mol['BLK_2'] = 'MOL_2'
        blk2mol['BLK_3'] = 'MOL_3'
        blk2mol['BLK_4'] = 'MOL_4'

        cut = {}
        cut['BLK_1'] = 506
        cut['BLK_2'] = 506
        cut['BLK_3'] = 400
        cut['BLK_4'] = 400

        p = {}
        p['BLK_1'] = 0.851
        p['BLK_2'] = 0.901
        blk_diffs = []

        for i, time in enumerate(df.index):
            month = time.month
            if month == 4:
                p['BLK_3'] = 0.710
                p['BLK_4'] = 0.700        
            elif month == 5:
                p['BLK_3'] = 0.742
                p['BLK_4'] = 0.732
            elif month == 6:
                p['BLK_3'] = 0.759
                p['BLK_4'] = 0.749
            else:
                p['BLK_3'] = 0.0
                p['BLK_4'] = 0.0
                
            if i == 0:
                df_stock.iloc[i] = df.iloc[i]    
            else:
                df_stock.iloc[i] = df_stock.iloc[i-1] + df.iloc[i]
                for column in df_order.columns:
                    val = df_order.loc[time, column]
                    if val > 0:
                        mol_col = blk2mol[column]
                        mol_num = df_stock.loc[time, mol_col]
                        df_stock.loc[time, mol_col] = 0     
                        
                        blk_gen = int(mol_num*p[column]*cut[column])
                        blk_stock = df_stock.loc[time, column] + blk_gen
                        blk_diff = blk_stock - val
                        
                        df_stock.loc[time, column] = blk_diff
                        print(f'{time}, {column}[{df_stock.loc[time, column]}]')
                        blk_diffs.append(blk_diff)

        return df_stock, blk_diffs    

    def add_stock(self, df, df_stock):
        df_out = df.copy()
        for column in df_out.columns:
            df_out.iloc[0][column] = df_out.iloc[0][column] + df_stock.iloc[0][column]
        return df_out

    def order_rescale(self, df, df_order):
        df_rescale = df.drop(df.columns, axis=1)
        dt = pd.Timedelta(hours=18)
        for column in ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']:
            for time in df_order.index:
                df_rescale.loc[time+dt, column] = df_order.loc[time, column]
        df_rescale = df_rescale.fillna(0)
        return df_rescale

    def cal_score(self, blk_diffs):
        # Block Order Difference
        blk_diff_m = 0
        for item in blk_diffs:
            if item < 0:
                blk_diff_m = blk_diff_m + abs(item)
        score = blk_diff_m
        return score

    # @tf.function
    def get_score(self, is_print=False):
        """
        p: 수요 발생 시 블럭 장난감 생산 부족분 합계
        q: 수요 발생 시 블럭 장난감 생산 초과분 합계
        c: 성형 공정 변경 시간 합계
        c_n: 성형 공정 변경 이벤트 횟수
        s: 멈춤 시간 합계
        s_n: 멈춤 이벤트 횟수
        N: 블럭 장난감 총 수요
        M: 전체 시간"""

        c = 0
        c_n = 0
        s = 0
        s_n = 0

        last_state = "None"
        for event_name in self.submission["Event_A"]:
            if "CHANGE" in event_name:
                c += 1
                if event_name != last_state:
                    c_n += 1
            elif event_name == "STOP":
                s += 1
                if event_name != last_state:
                    s_n += 1
            last_state = event_name

        last_state = "None"
        for event_name in self.submission["Event_B"]:
            if "CHANGE" in event_name:
                c += 1
                if event_name != last_state:
                    c_n += 1
            elif event_name == "STOP":
                s += 1
                if event_name != last_state:
                    s_n += 1
            last_state = event_name

        c /= 2
        s /= 2

        df = self.subprocess(self.submission) 

        out_1 = self.cal_schedule_part_1(df)
        out_2 = self.cal_schedule_part_2(df, line='A')
        out_3 = self.cal_schedule_part_2(df, line='B')
        out = out_1 + out_2 + out_3
        out = self.add_stock(out, self.stock)

        pd.set_option('display.max_rows', 10)
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.width', 1000)

        order = self.order_rescale(out, self.order)

        out, blk_diffs = self.cal_stock(out, order)
        score = self.cal_score(blk_diffs)

        N = order.to_numpy().sum()
        
        p = 0
        q = 0
        for diff in blk_diffs:
            if diff > 0:
                q += diff
            else:
                p -= diff

        
        M = out.shape[0]
        
        F1 = self.F(p, 10 * N)
        F2 = self.F(q, 10 * N)
        F3 = self.F(c, M) / (1+0.1 * c_n)
        F4 = self.F(s, M) /( 1+0.1 * s_n)
        Score = 50 * F1 + 20 * F2 + 20 * F3  + 10 * F4

        if is_print:
            print(f"N: {N}, p: {p}, q: {q}")
            print(f"M: {M}, c: {c}, c_n: {c_n}, s: {s}, s_n: {s_n}")

            print(f"F(p, 10N) : {F1}")
            print(f"F(q, 10N) : {F2}")
            print(f"F(c, M) / (1+0.1 x c_n) : {F3}")
            print(f"F(s, M) / (1+0.1 x s_n) : {F4}")
            
            print()
            print(f"Score : {Score}")

        return score

import timeit

if __name__ == "__main__":
    FILE_NAME = "submit.csv"
    submission = pd.read_csv(FILE_NAME)

    cal = Calculator(submission)
    cal.get_score(is_print=True)
    
    # cal.get_score(True)

