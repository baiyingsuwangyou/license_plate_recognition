import cv2
import os
import numpy as np
import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

# 模版匹配
template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '_chuan', '_e', '_gan0', '_gan1', '_gui0', '_gui1', '_hei', '_hu', '_ji0', '_ji1', '_jin0', '_jin1',
            '_jing', '_liao', '_lu', '_meng', '_min', '_ning', '_qing', '_qiong', '_shan', '_su', '_wan', '_xiang',
            '_xin', '_yu0', '_yu1', '_yue', '_yun', '_zhe', '_cang']

dict_template = {'_chuan': '川', '_e': '鄂', '_gan0': '甘', '_gan1': '赣', '_gui0': '贵', '_gui1': '桂', '_hei': '黑',
                 '_hu': '沪', '_ji0': '吉', '_ji1': '冀', '_jin0': '津', '_jin1': '晋', '_jing': '京', '_liao': '辽',
                 '_lu': '鲁', '_meng': '蒙', '_min': '闽', '_ning': '宁', '_qing': '青', '_qiong': '琼', '_shan': '陕',
                 '_su': '苏', '_wan': '皖', '_xiang': '湘', '_xin': '新', '_yu0': '渝', '_yu1': '豫', '_yue': '粤',
                 '_yun': '云', '_zhe': '浙', '_zang': '藏'}


def show_(time=-1, **img):
    """
    **img -> {winname1: img1, winname2: img2, ...}
    """
    for i in img:
        cv2.imshow(i, img[i])
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def read_dir(directory_name):
    """读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表"""
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list


class read_dir_list:
    def __init__(self, path):
        self.path = path

    def get_chinese_words_list_knn(self):
        """获得中文模板列表"""
        chinese_words_list = []
        label_chinese = []
        for i in range(34, 64):
            # 将模板存放在字典中
            c_words = read_dir(self.path + template[i])
            for c_word in c_words:
                chinese_words_list.append(c_word)
                label_chinese.append(i)
        return chinese_words_list, label_chinese

    def get_chinese_words_list_match(self):
        """获得中文模板列表"""
        chinese_words_list = []
        for i in range(34, 64):
            c_word = read_dir(self.path + template[i])
            chinese_words_list.append(c_word)
        return chinese_words_list

    def get_eng_words_list_match(self):
        """获得英文模板列表"""
        eng_words_list = []
        for i in range(10, 34):
            e_word = read_dir(self.path + template[i])
            eng_words_list.append(e_word)
        return eng_words_list

    def get_eng_num_words_list_match(self):
        """获得英文和数字模板列表"""
        eng_num_words_list = []
        for i in range(0, 34):
            word = read_dir(self.path + template[i])
            eng_num_words_list.append(word)
        return eng_num_words_list


def template_score(templateptah, image):
    """读取一个模板地址与图片进行匹配，返回得分"""
    # 读取模板图像
    template_img = cv2.imread(templateptah, 0)
    # 模板图像阈值化处理——获得黑白图
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)

    image_ = image.copy()
    # 获得待检测图片的尺寸
    height, width = image_.shape
    # 将模板resize至与图像一样大小
    template_img = cv2.resize(template_img, (width, height))
    # 模板匹配，返回匹配得分
    result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
    minval, maxval, maxloc, minloc = cv2.minMaxLoc(result)
    return maxval


class Cv_Car_blue(QWidget):
    def __init__(self):
        self.lab_picture2 = None
        self.lab_text = None
        super(Cv_Car_blue, self).__init__()
        self.main_UI()
        self.button_UI()

    def main_UI(self):
        self.setWindowTitle('main')  # 设置窗口名称
        self.setGeometry(100, 100, 1200, 800)  # 设置窗位置和大小

    def button_UI(self):
        btn_1 = QPushButton('退出', self)
        btn_2 = QPushButton('添加图片', self)
        btn_3 = QPushButton('识别车牌', self)

        # 设置位置和大小
        btn_1.setGeometry(900, 700, 200, 40)
        btn_2.setGeometry(500, 700, 200, 40)
        btn_3.setGeometry(100, 700, 200, 40)

        # 按钮链接函数
        btn_1.clicked.connect(self.close)
        btn_2.clicked.connect(self.Tow2)
        btn_3.clicked.connect(self.Tow3)

    def lab_text_UI(self, text, x, y, a, b, size=16, h=Qt.AlignmentFlag.AlignLeft, v=Qt.AlignmentFlag.AlignTop):
        self.lab_text = QLabel(text, self)
        self.lab_text.setWordWrap(True)
        self.lab_text.setFont(QFont('Arial', size, QFont.Weight.Bold))  # 设置字体
        self.lab_text.setGeometry(x, y, a, b)  # 设置位置和大小
        self.lab_text.setAlignment(h)  # 水平方向对齐
        self.lab_text.setAlignment(v)  # 垂直方向对齐
        self.lab_text.show()

    def lab_picture_UI1(self, x, y, a, b):
        self.lab_picture1 = QLabel(self)
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
        self.lab_picture1.setPixmap(QPixmap.fromImage(img))  # 在label上显示图片
        self.lab_picture1.setScaledContents(True)  # 让图片自适应label大小
        self.lab_picture1.setGeometry(x, y, a, b)
        self.lab_picture1.show()

    def lab_picture_UI2(self, x, y, a, b):
        self.lab_picture2 = QLabel(self)
        img = cv2.cvtColor(self.img_car, cv2.COLOR_BGR2RGB)
        img = QImage(img, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
        self.lab_picture2.setPixmap(QPixmap.fromImage(img))  # 在label上显示图片
        self.lab_picture2.setScaledContents(True)  # 让图片自适应label大小
        self.lab_picture2.setGeometry(x, y, a, b)
        self.lab_picture2.show()

    def Tow2(self):
        if self.lab_picture2 is not None and self.lab_text is not None:
            self.lab_picture2.hide()
            self.lab_text.hide()
        # 设置文件扩展名过滤,用双分号间隔
        filepath, filetype = QFileDialog.getOpenFileName(self, '选取图片', './image/',
                                                         'All Files (*);;PNG (*.png);;JPG (*.jpg;*.jpeg;*.jpe;*.jfif)')
        print(filepath)
        # self.img = cv2.imread(filepath, 1)
        self.img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), 1)
        self.lab_picture_UI1(450, 100, 700, 500)

    def Tow3(self):
        if self.lab_picture2 is not None and self.lab_text is not None:
            self.lab_text.hide()
            self.lab_picture2.hide()
        car_result = self.process()
        self.lab_text_UI(car_result, 100, 350, 300, 125, 24)
        self.lab_picture_UI2(50, 100, 350, 125)
        self.lab_text.show()
        self.lab_picture2.show()

    def closeEvent(self, event):
        ok = QPushButton()
        cancel = QPushButton()
        msg = QMessageBox(QMessageBox.Icon.Warning, '关闭', '是否关闭！')
        msg.addButton(ok, QMessageBox.ButtonRole.ActionRole)
        msg.addButton(cancel, QMessageBox.ButtonRole.RejectRole)
        ok.setText('确定')
        cancel.setText('取消')
        if msg.exec() == 1:
            event.ignore()
        else:
            # if self.cap.isOpened():
            #     self.cap.release()
            # if self.timer_camera.isActive():
            #     self.timer_camera.stop()
            event.accept()

    def process(self):
        """过程"""
        img_blue01 = self.get_bule01(self.img)
        # cv2.imshow('1', img_blue01)
        # cv2.waitKey()
        img_sobel = self.get_sobel()
        # 和
        img_ = cv2.bitwise_and(img_sobel, img_blue01)
        # cv2.imshow('1', img_)
        # cv2.waitKey()
        # 二值化并加大轮廓(准备提取轮廓)
        t, img_ = cv2.threshold(img_, 90, 255, cv2.THRESH_BINARY)
        kernel = np.ones(shape=(30, 30), dtype=np.uint8)
        img_ = cv2.morphologyEx(img_, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones(shape=(9, 9), dtype=np.uint8)
        img_ = cv2.erode(img_, kernel)
        img_ = cv2.dilate(img_, kernel)
        # cv2.imshow('1', img_)
        # cv2.waitKey()
        a = self.get_contour(img_)
        if a != '0':
            return a
        else:
            img_s1 = self.get_sobel1()
            img_ = cv2.bitwise_and(self.img, self.img, mask=img_s1)
            img_ = self.get_bule01(img_)
            kernel = np.ones(shape=(25, 25), dtype=np.uint8)
            img_ = cv2.morphologyEx(img_, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones(shape=(9, 9), dtype=np.uint8)
            img_ = cv2.erode(img_, kernel)
            img_ = cv2.dilate(img_, kernel)
            # cv2.imshow('1', img_)
            # cv2.waitKey()
            self.get_contour(img_)

    def get_bule01(self, img):
        """提取车牌(蓝色)区域"""
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        t, s = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow('1', s)
        # cv2.waitKey()
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        img_ = cv2.bitwise_and(img, s)
        lower_blue = np.array([80, 0, 0])
        upper_blue = np.array([225, 200, 120])
        mask = cv2.inRange(img_, lower_blue, upper_blue)
        img_blue = cv2.bitwise_and(img, img, mask=mask)
        # 颜色二值化
        img_blue = cv2.cvtColor(img_blue, cv2.COLOR_BGR2GRAY)
        t, img_blue01 = cv2.threshold(img_blue, 80, 255, cv2.THRESH_BINARY)
        # cv2.imshow('1', img_blue01)
        # cv2.waitKey()
        return img_blue01

    def get_sobel(self):
        """sobel边缘检测"""
        img_Gau = cv2.GaussianBlur(self.img, (3, 3), 0, 0)
        img_grey = cv2.cvtColor(img_Gau, cv2.COLOR_BGR2GRAY)
        img_sobel = cv2.convertScaleAbs(cv2.Sobel(img_grey, cv2.CV_64F, 1, 0, 5))
        kernel = np.ones(shape=(30, 30), dtype=np.uint8)
        img_close = cv2.morphologyEx(img_sobel, cv2.MORPH_CLOSE, kernel)
        return img_close

    def get_sobel1(self):
        img_Gau = cv2.GaussianBlur(self.img, (9, 9), 0, 0)
        img_grey = cv2.cvtColor(img_Gau, cv2.COLOR_BGR2GRAY)
        img_sobel = cv2.convertScaleAbs(cv2.Sobel(img_grey, cv2.CV_64F, 1, 0))
        kernel = np.ones(shape=(15, 15), dtype=np.uint8)
        img_ = cv2.morphologyEx(img_sobel, cv2.MORPH_CLOSE, kernel)
        img_ = cv2.morphologyEx(img_, cv2.MORPH_OPEN, kernel)
        t, img_ = cv2.threshold(img_, 50, 255, cv2.THRESH_BINARY)
        return img_

    def get_BlueImg_bin(self):
        # 掩膜：BGR通道，若像素B分量在 100~255 且 G分量在 0~190 且 G分量在 0~140 置255（白色） ，否则置0（黑色）
        mask_gbr = cv2.inRange(self.img, (100, 0, 0), (255, 190, 140))

        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)  # 转换成 HSV 颜色空间
        h, s, v = cv2.split(img_hsv)  # 分离通道  色调(H)，饱和度(S)，明度(V)
        mask_s = cv2.inRange(s, 80, 255)  # 取饱和度通道进行掩膜得到二值图像

        rgbs = mask_gbr & mask_s  # 与操作，两个二值图像都为白色才保留，否则置黑
        # 核的横向分量大，使车牌数字尽量连在一起
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))
        img_rgbs_dilate = cv2.dilate(rgbs, kernel, 3)  # 膨胀 ，减小车牌空洞
        return img_rgbs_dilate

    def get_contour(self, img_):
        """轮廓提取"""
        contours, hierarchy = cv2.findContours(img_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print('findContours:', len(contours))
        # 排除无效轮廓
        boxs = []
        minAreaRects = []
        contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            area_width, area_height = rect[1]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            if area_height == 0:
                continue
            wh_ratio = area_width / area_height
            if 2 < wh_ratio < 5.5:
                minAreaRects.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                boxs.append(box)

        print('minAreaRects:', len(minAreaRects))
        if len(minAreaRects) == 0:
            print('minAreaRects=0!!!')
            return '0'

        # 矫正角度
        center, size, angle = minAreaRects[0][0], minAreaRects[0][1], minAreaRects[0][2]
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # print(angle)
        if size[0] < size[1]:
            w = size[1]
            h = size[0]
            angle -= 90
            size = (w, h)
        height, width = self.img.shape[0], self.img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(self.img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
        w, h = img_crop.shape[:2]
        self.img_car = cv2.resize(img_crop, dsize=(200, int(200 * w / h)))

        # 二值化车牌图像
        img_car_gray = cv2.cvtColor(self.img_car, cv2.COLOR_BGR2GRAY)
        t, img_car01 = cv2.threshold(img_car_gray, 127, 255, cv2.THRESH_BINARY)

        # 在原图中框出车牌
        # img = cv2.drawContours(img, boxs, -1, (0, 0, 255), 2)
        # show_(img=img, img_car=img_car01)

        n = 5
        useless = True
        for j in range(n):
            # 膨胀或连接
            if useless:
                ke = 7 - j
                kernel = np.ones(shape=(ke, ke), dtype=np.uint8)
                img_car01_ = cv2.morphologyEx(img_car01, cv2.MORPH_CLOSE, kernel)
                useless = False
            else:
                ke = 6 - j
                kernel = np.ones(shape=(ke, ke), dtype=np.uint8)
                img_car01_ = cv2.dilate(img_car01, kernel)
                useless = True
            img_car01_[0:3, :] = 0
            img_car01_[-3:, :] = 0
            img_car01_[:, 0:4] = 0
            img_car01_[:, -2:] = 0
            # cv2.imshow('1', img_car01_)
            # cv2.imshow('2', img_car01)
            # cv2.waitKey(4000)
            # 查找轮廓
            contours, hierarchy = cv2.findContours(img_car01_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            words = []
            word_images = []
            # 对所有轮廓逐一操作
            for contour in contours:
                word = []
                rect = cv2.boundingRect(contour)
                x = rect[0]
                y = rect[1]
                weight = rect[2]
                height = rect[3]
                word.append(x)
                word.append(y)
                word.append(weight)
                word.append(height)
                words.append(word)
            # 排序，车牌号有顺序。words是一个嵌套列表
            words = sorted(words, key=lambda x: x[0], reverse=False)
            print('words:', words, len(words))
            i = 0
            print('words index:', end=' ')
            # word中存放轮廓的起始点和宽高
            for word in words:
                # 筛选字符的轮廓
                if (word[2] * 1.5) < word[3] < (word[2] * 3.5) or (word[2] >= 5 and word[3] >= 25):
                    i += 1
                    print(words.index(word), end=' ')
                    if i == 1:
                        splite_image = img_car01[0:word[1] + word[3], word[0]:word[0] + word[2]]
                    else:
                        if word[1] != 0:
                            splite_image = img_car01[word[1] - 1:word[1] + word[3], word[0] - 1:word[0] + word[2] + 1]
                        else:
                            splite_image = img_car01[word[1]:word[1] + word[3], word[0] - 1:word[0] + word[2]]
                    # cv2.imshow('1', splite_image)
                    # cv2.waitKey(1000)
                    word_images.append(splite_image)

            print('word_images:', len(word_images))
            if len(word_images) == 7:
                print('the number is enough')
                break
            if j == n - 1:
                return '0'
        # 获取匹配模板
        read_template = read_dir_list('./data/')
        chinese_words_list_knn, label_chinese = read_template.get_chinese_words_list_knn()
        chinese_words_list_match = read_template.get_chinese_words_list_match()
        eng_words_list_match = read_template.get_eng_words_list_match()
        eng_num_words_list_match = read_template.get_eng_num_words_list_match()

        template_row = 20
        template_col = 20  # 模板图片的尺寸

        results = []
        for index, word_image in enumerate(word_images):
            word_image = cv2.resize(word_image, dsize=(template_col, template_row), )
            # 匹配中文字符(1)
            if index == 0:
                # knn
                n = 0
                a = np.zeros(shape=(len(chinese_words_list_knn), template_row, template_col))
                for path in chinese_words_list_knn:
                    a[n, :, :] = cv2.imread(path, 0)
                    n += 1
                feature = np.zeros(
                    shape=(len(chinese_words_list_knn), round(template_row / 5), round(template_col / 5)))
                for i in range(len(chinese_words_list_knn)):
                    for j in range(template_row):
                        for k in range(template_col):
                            if a[i, j, k] == 255:
                                feature[i, int(j / 5), int(k / 5)] += 1
                train = feature[:, :].reshape(-1, round(template_row / 5) * round(template_col / 5)).astype(
                    np.float32)
                train_label = np.asarray(label_chinese)
                of = np.zeros(shape=(round(template_row / 5), round(template_col / 5)))
                for i in range(template_row):
                    for j in range(template_col):
                        if word_image[i, j] == 255:
                            of[int(i / 5), int(j / 5)] += 1
                test = of.reshape(-1, round(template_row / 5) * round(template_col / 5)).astype(np.float32)
                knn = cv2.ml.KNearest_create()
                knn.train(train, cv2.ml.ROW_SAMPLE, train_label)
                ret, result, neighbours, dist = knn.findNearest(test, k=5)
                r_knn = template[int(result[0, 0])]
                # print(r_knn, result, neighbours, dist)
                # template_match
                best_score = []
                for chinese_words in chinese_words_list_match:
                    score = []
                    for chinese_word in chinese_words:
                        result = template_score(chinese_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                r_match = template[34 + i]

                print('r_match:', r_match, 'r_knn:', r_knn)
            # 匹配英文字母(2)
            elif index == 1:
                best_score = []
                for eng_word_list in eng_words_list_match:
                    score = []
                    for eng_word in eng_word_list:
                        result = template_score(eng_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                r = template[10 + i]
                results.append(r)
                print('english word is complete!!! ', r)
            # 匹配字母和数字(3~7)
            else:
                best_score = []
                for eng_num_word_list in eng_num_words_list_match:
                    score = []
                    # list_ = []  #
                    for eng_num_word in eng_num_word_list:
                        result = template_score(eng_num_word, word_image)
                        score.append(result)
                        # list_.append(eng_num_word)  #
                    best_score.append(max(score))
                    # index_ = score.index(max(score))  #
                    # a = list_[index_]  #
                i = best_score.index(max(best_score))
                r = template[i]
                results.append(r)
                print('english word or number is complete!!! ', r)

        if r_knn == r_match:
            result = dict_template[r_match] + ''.join(results)
        else:
            result = dict_template[r_match] + ''.join(results)
        return result


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Cv_Car_blue()
    win.show()
    sys.exit(app.exec())

