import random
import os.path as osp
import os
import pandas as pd
import numpy as np
# 朴素贝叶斯处理泰坦尼克号数据集
# 特征向量选取('Pclass','Sex‘， ’Age‘, 'SibSp', 'Parch, 'Embarked')
# Age 和 Embarked需要进行数据清洗。只有age是连续量，需要先离散化
# 训练集的统计信息都被存入NBTitannic的P_Y和P_X_Y属性中

dataset_root = 'dataset/week2'

class NBTitanic:
    def __init__(self, dataset_root):
        self.train_csv = pd.read_csv(osp.join(dataset_root, 'train.csv'))
        self.test_csv = pd.read_csv(osp.join(dataset_root, 'test.csv'))
        self.avg_age_dict = self.genAvgAge()
        self.age_stage = (6, 12, 17, 45, 69, 300)

        self.test_start = 892
        self.train_Larray = self.normalize(self.clean(self.train_csv), istrain=True)
        self.test_Larray = self.normalize(self.clean(self.test_csv))
        self.feature_key = ('Pclass', 'Sex', 'Age', 'Sibsp', 'Parch', 'Embarked')
        self.pcdStatisticsInfo()
        print('')

    def genAvgAge(self):
        pClass_1_ages = []
        pClass_2_ages = []
        pClass_3_ages = []
        for index, row in self.train_csv.iterrows():
            pclass = row['Pclass']
            age = row['Age']
            if pclass == 1 and not np.isnan(age):
                pClass_1_ages.append(age)
            elif pclass == 2 and not np.isnan(age):
                pClass_2_ages.append(age)
            elif pclass == 3 and not np.isnan(age):
                pClass_3_ages.append(age)
        avg_age_dict = {
            1:np.mean(pClass_1_ages),
            2:np.mean(pClass_2_ages),
            3:np.mean(pClass_3_ages)
        }
        return avg_age_dict

    def clean(self, data_frame):
        for index, row in data_frame.iterrows():
            if np.isnan(row['Age']):
                pclass = row['Pclass']
                avg_age = self.avg_age_dict[pclass]
                data_frame.loc[index, 'Age'] = avg_age
            embarked = row['Embarked']
            if pd.isna(embarked):
                data_frame.loc[index, 'Embarked'] = random.choice(['S', 'C', 'Q'])
        return data_frame

    def normalize(self, data_frame, istrain=False):
        farray = []

        for index, row in data_frame.iterrows():
            pclass, sex, age, sibsp, parch, embarked = \
            row['Pclass'], row['Sex'], row['Age'], row['SibSp'], row['Parch'], row['Embarked']

            pclass = pclass - 1

            if sex == 'male':
                sex = 0
            elif sex == 'female':
                sex = 1
            else:
                assert 0

            for n, age_stage in enumerate(self.age_stage):
                if age <= age_stage:
                    age = n
                    break

            sibsp = sibsp
            parch = parch

            if embarked == 'S':
                embarked = 0
            elif embarked == 'C':
                embarked = 1
            elif embarked == 'Q':
                embarked = 2
            else:
                assert 0

            if istrain:
                survived = row['Survived']
                feature_list = [pclass, sex, age, sibsp, parch, embarked, survived]
            else:
                feature_list = [pclass, sex, age, sibsp, parch, embarked]
            farray.append(feature_list)

        return farray

    def pcdStatisticsInfo(self):
        self.P_Y = {}
        np_array = np.array(self.train_Larray, dtype='uint8')
        self.x_max = np.max(np_array, axis=0)[:-1]
        label = np_array[:,-1].tolist()
        feature_Larray = np_array[:,:-1].tolist()
        p_y_0, p_y_1 = 0, 0
        for l in label:
            if l == 1:
                p_y_1 += 1
            else:
                p_y_0 += 1
        self.P_Y[0] = p_y_0 / len(label)
        self.P_Y[1] = p_y_1 / len(label)


        Y_len = 2
        F_len = len(self.feature_key)
        X_len = np.max(feature_Larray) + 1
        self.P_X_Y = np.ones((Y_len, F_len, X_len), dtype='float32')
        len_array = np.ones((Y_len,F_len), dtype='float32')
        for f_vector, y_idx in zip(feature_Larray, label):
            for f_idx, x_idx in enumerate(f_vector):
                self.P_X_Y[y_idx,f_idx, x_idx] += 1
                len_array[y_idx, f_idx] += 1

        len_array = len_array[:,:,np.newaxis]
        self.P_X_Y = self.P_X_Y / len_array

    def pred(self, feature_vector):
        y_0 = self.P_Y[0]
        y_1 = self.P_Y[1]

        for f_idx, x_idx in enumerate(feature_vector):
            if x_idx > self.x_max[f_idx]:
                x_idx = self.x_max[f_idx]
            y_0 *= self.P_X_Y[0, f_idx, x_idx]
            y_1 *= self.P_X_Y[1, f_idx, x_idx]
        # y_0_ = y_0 / (y_0 + y_1)
        # y_1_ = y_1 / (y_0 + y_1)
        #
        # print(f'feature:{feature_vector}\ny_0:{y_0_:>.6f}\ty_1:{y_1_:>.6f}')
        if y_1 > y_0:
            op = 1
        else:
            op = 0
        return op

    def __call__(self):
        data = []
        for PassengerId, f_v in enumerate(self.test_Larray, start=self.test_start):
            Survived = self.pred(f_v)
            data.append([PassengerId, Survived])
        columns = ['PassengerId', 'Survived']
        csv = pd.DataFrame(data, columns=columns)
        save_root = osp.join('output', 'week2')
        if not osp.exists(save_root):
            os.makedirs(save_root)
        csv.to_csv(osp.join(save_root, 'submission.csv'), index=False)

if __name__ == '__main__':
    nb = NBTitanic(dataset_root)
    nb()


