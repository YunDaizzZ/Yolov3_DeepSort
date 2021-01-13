# coding:utf-8
from __future__ import division
import numpy as np
import collections
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

class Hungarian():
    def __init__(self, input_matrix=None, is_profit_matrix=False):
        # 输入为一个二维嵌套列表
        # is_profit_matrix=False代表输入是消费矩阵（需要使消费最小化），反之则为利益矩阵（需要使利益最大化）
        if input_matrix is not None:
            # 保存输入
            my_matrix = np.array(input_matrix)
            self._input_matrix = np.array(input_matrix)
            self._maxCol = my_matrix.shape[1]
            self._maxRow = my_matrix.shape[0]

            # 本算法作用于方阵，若不是则填充0变为方阵
            matrix_size = max(self._maxCol, self._maxRow)
            pad_cols = matrix_size - self._maxRow
            pad_rows = matrix_size - self._maxCol
            my_matrix = np.pad(my_matrix, ((0, pad_cols), (0, pad_rows)), 'constant', constant_values=(0))

            # 如果需要，则转化为消费矩阵
            if is_profit_matrix:
                my_matrix = self.make_cost_matrix(my_matrix)

            self._cost_matrix = my_matrix
            self._size = len(my_matrix)
            self._shape = my_matrix.shape

            # 存放算法结果
            self._results = []
            self._totalPotential = 0
        else:
            self._cost_matrix = None

    def make_cost_matrix(self, profit_matrix):
        # 利益矩阵转化为消费矩阵
        matrix_shape = profit_matrix.shape
        offset_matrix = np.ones(matrix_shape, dtype=int) * profit_matrix.max()
        cost_matrix = offset_matrix - profit_matrix

        return cost_matrix

    def get_results(self):
        # 获取算法结果
        return self._results

    def calculate(self):
        # 实施匈牙利算法
        result_matrix = self._cost_matrix.copy()

        # 步骤1：矩阵每一行减去本行最小值
        for index, row in enumerate(result_matrix):
            result_matrix[index] -= row.min()

        # 步骤2：矩阵每一列减去本列最小值
        for index, col in enumerate(result_matrix.T):
            result_matrix[:, index] -= col.min()

        # 步骤3：使用最少数量的划线覆盖矩阵中所有的0元素
        # 如果划线总数不等于矩阵的维度需要进行矩阵调整并重复循环此步骤
        total_covered = 0
        while total_covered < self._size:
            # 使用最少数量的划线覆盖矩阵中所有的0元素同时记录划线数量
            cover_zeros = CoverZeros(result_matrix)
            single_zero_pos_list = cover_zeros.calculate()
            covered_rows = cover_zeros.get_covered_rows()
            covered_cols = cover_zeros.get_covered_cols()
            total_covered = len(covered_rows) + len(covered_cols)

            # 如果划线总数不等于矩阵维度需要进行矩阵调整（需要使用未覆盖处的最小元素）
            if total_covered < self._size:
                result_matrix = self._adjust_matrix_by_min_uncovered_num(result_matrix, covered_rows, covered_cols)
        # 元组形式结果对存放到列表
        self._results = single_zero_pos_list
        # 计算总期望结果
        value = 0
        for row, col in single_zero_pos_list:
            value += self._input_matrix[row, col]
        self._totalPotential = value

    def get_total_potential(self):
        return self._totalPotential

    def _adjust_matrix_by_min_uncovered_num(self, result_matrix, covered_rows, covered_cols):
        # 计算未覆盖元素中的最小值m，未覆盖元素减去m，行列划线交叉处加上m
        adjusted_matrix = result_matrix
        # 计算未覆盖元素中的最小值m
        elements = []
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_cols:
                        elements.append(element)
        min_uncovered_num = min(elements)
        # 未覆盖元素减去m
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_cols:
                        adjusted_matrix[row_index, index] -= min_uncovered_num

        # 行列划线交叉处加上m
        for row_ in covered_rows:
            for col_ in covered_cols:
                adjusted_matrix[row_, col_] += min_uncovered_num

        return adjusted_matrix

class CoverZeros():
    # 使用最少数量的划线覆盖矩阵中的所有零
    def __init__(self, matrix):
        # 找到矩阵中零的位置（0为True，非0为False）
        self._zero_locations = (matrix == 0)
        self._zero_locations_copy = self._zero_locations.copy()
        self._shape = matrix.shape

        # 存储划线盖住的行列
        self._covered_rows = []
        self._covered_cols = []

    def get_covered_rows(self):
        # 返回覆盖行索引列表
        return self._covered_rows

    def get_covered_cols(self):
        # 返回覆盖列索引列表
        return self._covered_cols

    def row_scan(self, marked_zeros):
        # 扫描矩阵每一行，找到含0元素最少的行，对任意0元素标记（独立0元素），
        # 划去标记0元素（独立0元素）所在行和列存在的0元素
        min_row_zero_nums = [9999999, -1]
        for index, row in enumerate(self._zero_locations_copy):  # index为行号
            row_zero_nums = collections.Counter(row)[True]
            if row_zero_nums < min_row_zero_nums[0] and row_zero_nums != 0:
                # 找最少0元素的行
                min_row_zero_nums = [row_zero_nums, index]
        # 最少0元素的行
        row_min = self._zero_locations_copy[min_row_zero_nums[1], :]
        # 找到此行中任意一个0元素的索引位置即可
        row_indices, = np.where(row_min)
        # 标记该0元素
        marked_zeros.append((min_row_zero_nums[1], row_indices[0]))
        # 划去该0元素所在行和列所在的0元素
        # 因为被覆盖，所以把二值矩阵_zero_locations中相应的行列全部置为False
        self._zero_locations_copy[:, row_indices[0]] = np.array([False for _ in range(self._shape[0])])
        self._zero_locations_copy[min_row_zero_nums[1], :] = np.array([False for _ in range(self._shape[0])])

    def calculate(self):
        # 进行计算
        # 存储勾选的行和列
        ticked_row = []
        ticked_col = []
        marked_zeros = []
        # 1 试指派并标记独立0元素
        while True:
            # 循环直到所有0元素被处理（_zero_locations中没有True）
            if True not in self._zero_locations_copy:
                break
            self.row_scan(marked_zeros)

        # 2 无被标记0（独立0元素）的行打钩
        independent_zero_row_list = [pos[0] for pos in marked_zeros]
        ticked_row = list(set(range(self._shape[0])) - set(independent_zero_row_list))
        # 重复3 4直到不能在打钩
        TICK_FLAG = True
        while TICK_FLAG:
            TICK_FLAG = False
            # 3 对打钩的行中所含0元素的列打钩
            for row in ticked_row:
                # 找到此行
                row_array = self._zero_locations[row, :]
                # 找到此行中0元素的索引位置
                for i in range(len(row_array)):
                    if row_array[i] == True and i not in ticked_col:
                        ticked_col.append(i)
                        TICK_FLAG = True

            # 4 对打钩的列中所含独立0元素的行打钩
            for row, col in marked_zeros:
                if col in ticked_col and row not in ticked_row:
                    ticked_row.append(row)
                    TICK_FLAG = True

        # 对打钩的列和没有打钩的行划线
        self._covered_rows = list(set(range(self._shape[0])) - set(ticked_row))
        self._covered_cols = ticked_col

        return marked_zeros

if __name__ == "__main__":
    test = np.array([[9, 11, 14, 11, 7],
                     [6, 15, 13, 13, 10],
                     [12, 13, 6, 8, 8],
                     [11, 9, 10, 12, 9],
                     [7, 12, 14, 10, 14]])
    result = Hungarian(test, is_profit_matrix=False)
    result.calculate()
    print result.get_results()
    print '----'
    # 两种自带的匈牙利算法函数
    print linear_assignment(test)
    print '----'
    print linear_sum_assignment(test)