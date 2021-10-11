
import numpy as np
from tqdm import tqdm
import os.path as osp
import os
import pandas as pd

# PCA主成分分析加朴素贝叶斯判断手写数字
class PCANBDigitalNum:
    def __init__(self, dataset_root='./dataset/week3', pca_length = 20, cls_num=10):
        assert osp.exists(dataset_root)
        self.dataset_root = dataset_root
        assert osp.exists(osp.join(self.dataset_root, 'train.csv'))
        assert osp.exists(osp.join(self.dataset_root, 'test.csv'))
        train_csv = pd.read_csv(osp.join(self.dataset_root, 'train.csv'))
        test_csv = pd.read_csv(osp.join(self.dataset_root, 'test.csv'))
        self.test_feMatrix = self.getRawFEMatrix(test_csv, is_train=False)
        self.train_feMatrix, _ = self.getRawFEMatrix(train_csv, is_train=True)
        train_feMatrix, train_label = self.getRawFEMatrix(train_csv, is_train=True)
        self.pca_lenght = pca_length
        self.cls_num = cls_num

        # train
        train_feMatrix, self.m_eigen = self.center(train_feMatrix)
        self.pca_matrix = self.getPCA(train_feMatrix, self.pca_lenght)
        train_feMatrix = self.pca_matrix.transpose() @ train_feMatrix
        # cls_num分类 pca_length 个特征
        self.mean_matrix, self.var_matrix, self.prior = self.train(train_label, train_feMatrix)

        print('')
    def getRawFEMatrix(self, csv, is_train=True):
        # 获取原始的特征向量矩阵
        feMatrix = csv.to_numpy(dtype='float64')
        if is_train:
            label = feMatrix[:, 0]
            feMatrix = feMatrix[:, 1:]
            # feMatrix_ = []
            # for v in feMatrix:
            #     v = v.reshape(28,28).flatten('F')
            #     feMatrix_.append(v)
            # feMatrix_ = np.array(feMatrix_)
            # feMatrix = np.concatenate([feMatrix, feMatrix_], 1)
            feMatrix = (feMatrix.transpose() > 125).astype('float32')
            return feMatrix, label
        else:
            # feMatrix_ = []
            # for v in feMatrix:
            #     v = v.reshape(28, 28).flatten('F')
            #     feMatrix_.append(v)
            # feMatrix_ = np.array(feMatrix_)
            # feMatrix = np.concatenate([feMatrix, feMatrix_], 1)
            feMatrix = (np.transpose(feMatrix) > 125).astype('float32')
            return feMatrix

    def center(self, feMatrix, m_eigen=None):
        if type(m_eigen) == type(None):
            m_eigen = np.mean(feMatrix, axis=1, keepdims=True)
            feMatrix = feMatrix - m_eigen
            return feMatrix, m_eigen
        else:
            return feMatrix - m_eigen


    def getPCA(self, feMtraix, length=11):
        cov = feMtraix @ np.transpose(feMtraix)
        values, vector_matrix = np.linalg.eigh(cov)
        vector_matrix = np.transpose(vector_matrix).tolist()
        values = values.tolist()
        # sort
        z = list(zip(values, vector_matrix))
        z.sort(key=lambda x: x[0], reverse=True)
        values, vector_matrix = zip(*z)
        v_sum = sum(values)
        if length == None:
            length = 0
            v_sum_ = 0
            for v in values:

                v_sum_ += v
                if v_sum_ / v_sum >= 0.95:
                    print(length)
                    break
                length += 1
        pca_matrix = np.array(vector_matrix[:length]).transpose()

        return pca_matrix

    def train(self, label, feMatrix):
        mean_matrix = []
        var_matrix = []
        prior = np.zeros((self.cls_num), dtype='float32')
        feMatrix = np.transpose(feMatrix).tolist()
        label = label.tolist()
        store = [[] for _ in range(self.cls_num)]
        for l, v in zip(label, feMatrix):
            l = int(l)
            store[l].append(v)
            prior[l] += 1
        for s in store:
            # (N, pca_length)
            s = np.array(s)
            mean = np.mean(s, axis=0)
            s = s - mean
            co_var = s.transpose() @ s
            mean_matrix.append(mean)
            var_matrix.append(co_var)
        prior /= float(len(label))
        return mean_matrix, var_matrix, prior.tolist()

    def loglikehood(self, mean, co_var, ip_vector):
        '''求符合高斯分布的似然，为防止下溢出，使用log型的
        Arg:
            mean\ip_vector: numpy array (N,)
            co_var: numpy array (N,N)
        '''
        n = len(ip_vector)
        mean = mean[:, np.newaxis]
        ip_vector = ip_vector[:, np.newaxis]
        inv_co_var = np.linalg.inv(co_var)
        det_co_var = np.linalg.det(co_var)
        res_v = ip_vector - mean
        like_hood = - np.log(det_co_var) - (res_v.transpose() @ inv_co_var @ res_v)[0][0]
        return like_hood

    def test(self):
        save_root = 'output/week3'
        if not osp.exists(save_root):
            os.makedirs(save_root)
        feMatrix = self.center(self.test_feMatrix, self.m_eigen)
        feMatrix = (self.pca_matrix.transpose() @ feMatrix).transpose()
        bar = tqdm(enumerate(feMatrix, start=1), total=len(feMatrix))
        output = []
        for ImageId, vector in bar:
            like_hoods = []
            for c in range(self.cls_num):
                vars = self.var_matrix[c]
                means = self.mean_matrix[c]
                like_hood = self.loglikehood(means, vars, vector)
                like_hoods.append(like_hood)
            like_hoods = np.array(like_hoods)
            # prior = np.log(np.array(self.prior))
            soft_one_hot = like_hoods
            Label = np.argmax(soft_one_hot, axis=0)
            output.append([ImageId, Label])
        columns = ['ImageId', 'Label']
        save_path = osp.join(save_root, 'submission.csv')
        csv = pd.DataFrame(output, columns=columns)
        csv.to_csv(save_path, index=False)

    def showTestPic(self):
        import cv2 as cv
        feMatrix = self.test_feMatrix.transpose()
        for v in feMatrix:
            v = v.reshape(28,28).astype('uint8')
            cv.imshow('0', v)
            cv.waitKey(0)
            cv.destroyAllWindows()

# PCA主成分分析加最小二乘判断手写数字
class PCALeastSquareDigitalNum:
    def __init__(self, dataset_root='./dataset/week3', pca_length = None, cls_num=10):
        assert osp.exists(dataset_root)
        self.dataset_root = dataset_root
        assert osp.exists(osp.join(self.dataset_root, 'train.csv'))
        assert osp.exists(osp.join(self.dataset_root, 'test.csv'))
        train_csv = pd.read_csv(osp.join(self.dataset_root, 'train.csv'))
        test_csv = pd.read_csv(osp.join(self.dataset_root, 'test.csv'))
        self.test_feMatrix = self.getRawFEMatrix(test_csv, is_train=False)
        self.train_feMatrix, _ = self.getRawFEMatrix(train_csv, is_train=True)
        train_feMatrix, train_label = self.getRawFEMatrix(train_csv, is_train=True)
        self.pca_lenght = pca_length
        self.cls_num = cls_num

        # train
        train_feMatrix, self.m_eigen = self.center(train_feMatrix)
        self.pca_matrix = self.getPCA(train_feMatrix, self.pca_lenght)
        train_feMatrix = self.pca_matrix.transpose() @ train_feMatrix
        self.A_, self.B_ = self.trainLeastSquare(train_feMatrix, train_label)


        print('done train')
    def getRawFEMatrix(self, csv, is_train=True):
        # 获取原始的特征向量矩阵
        feMatrix = csv.to_numpy(dtype='float64')
        if is_train:
            label = feMatrix[:, 0]
            feMatrix = feMatrix[:, 1:]
            # feMatrix_ = []
            # for v in feMatrix:
            #     v = v.reshape(28,28).flatten('F')
            #     feMatrix_.append(v)
            # feMatrix_ = np.array(feMatrix_)
            # feMatrix = np.concatenate([feMatrix, feMatrix_], 1)
            feMatrix = feMatrix.transpose() / 255
            return feMatrix, label
        else:
            # feMatrix_ = []
            # for v in feMatrix:
            #     v = v.reshape(28, 28).flatten('F')
            #     feMatrix_.append(v)
            # feMatrix_ = np.array(feMatrix_)
            # feMatrix = np.concatenate([feMatrix, feMatrix_], 1)
            feMatrix = np.transpose(feMatrix) / 255
            return feMatrix

    def center(self, feMatrix, m_eigen=None):
        if type(m_eigen) == type(None):
            m_eigen = np.mean(feMatrix, axis=1, keepdims=True)
            feMatrix = feMatrix - m_eigen
            return feMatrix, m_eigen
        else:
            return feMatrix - m_eigen


    def getPCA(self, feMtraix, length=11):
        cov = feMtraix @ np.transpose(feMtraix)
        values, vector_matrix = np.linalg.eigh(cov)
        vector_matrix = np.transpose(vector_matrix).tolist()
        values = values.tolist()
        # sort
        z = list(zip(values, vector_matrix))
        z.sort(key=lambda x: x[0], reverse=True)
        values, vector_matrix = zip(*z)
        v_sum = sum(values)
        if length == None:
            length = 0
            v_sum_ = 0
            for v in values:

                v_sum_ += v
                if v_sum_ / v_sum >= 0.995:
                    print(length)
                    break
                length += 1
        pca_matrix = np.array(vector_matrix[:length]).transpose()

        return pca_matrix

    def trainLeastSquare(self, feMatrix, label):
        f_num, n = feMatrix.shape
        X = feMatrix.transpose()
        X_ = np.concatenate([np.ones([n, 1], dtype=X.dtype), X], 1)
        X_T_ = np.transpose(X_)
        C_ = np.linalg.inv(X_T_ @ X_) @ X_T_ @ label[:,np.newaxis]
        B_ = C_[0]
        A_ = C_[1:]

        return A_, B_

    def pred(self, vector):
        vector = vector[:, np.newaxis]
        pred = vector.transpose() @ self.A_ + self.B_
        return np.clip(int(np.round(pred[0][0])), 0, 9)

    def test(self):
        save_root = 'output/week3'
        if not osp.exists(save_root):
            os.makedirs(save_root)
        feMatrix = self.center(self.test_feMatrix, self.m_eigen)
        feMatrix = (self.pca_matrix.transpose() @ feMatrix).transpose()
        bar = tqdm(enumerate(feMatrix, start=1), total=len(feMatrix))
        output = []
        for ImageId, vector in bar:
            Label = self.pred(vector)
            output.append([ImageId, Label])
        columns = ['ImageId', 'Label']
        save_path = osp.join(save_root, 'submission.csv')
        csv = pd.DataFrame(output, columns=columns)
        csv.to_csv(save_path, index=False)


if __name__ == '__main__':
    # pca_nb = PCANBDigitalNum()
    # pca_nb.test()
    pca_ls = PCALeastSquareDigitalNum()
    pca_ls.test()