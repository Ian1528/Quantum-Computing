import os
import sys
from ctypes import *

import PySide6
import newtrial
from PySide6 import QtWidgets
from PySide6.QtCore import QDir
from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor
from PySide6.QtWidgets import QFileDialog, QTabWidget

# UDP接收线程
from UDPThread import udppthread
# UI界面
from ui_main import Ui_MainWindow

class qtwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(qtwindow, self).__init__()
        self.setupUi(self)

        # 初始化界面
        self.lineEdit_ZJ_IP.setText("192.168.1.50")
        self.lineEdit_ZJ_Port.setText("1234")
        self.lineEdit_XWJ_IP.setText("192.168.1.10")
        self.lineEdit_XWJ_Port.setText("1234")
        self.lineEdit_Param_Delay.setText("1")
        self.lineEdit_Param_Gray.setText("1")
        self.lineEdit_Param_CFPicnum.setText("1")
        self.lineEdit_Param_CFDelay.setText("0")

        # 动态库
        path1 = os.getcwd()
        lib_path = path1 + "\\libFL_DLP_HS.dll"
        self.lib = windll.LoadLibrary(lib_path)
        #   类变量
        self.Device_ID = 0
        self.Thread_receive = None
        self.Pic_AddrTwo = None
        self.Pic_AddrEight = None
        self.Pic_AddrBin = None
        # 光标
        self.user_cursor = self.textEdit_User.textCursor()
        self.user_cursor.movePosition(QTextCursor.End)
        self.textEdit_User.setTextCursor(self.user_cursor)

        self.feedback_cursor = self.textEdit_FeedBack.textCursor()
        self.feedback_cursor.movePosition(QTextCursor.End)
        self.textEdit_FeedBack.setTextCursor(self.feedback_cursor)

        # 序列表格初始化
        self.tableWidget.setRowCount(8192)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setStyleSheet(
            "QHeaderView::section{"
            "border-top:0px solid #E5E5E5;"
            "border-left:0px solid #E5E5E5;"
            "border-right:0.5px solid #E5E5E5;"
            "border-bottom: 0.5px solid #E5E5E5;"
            "background-color:white;"
            "padding:0.5px;"
            "}"
        )
        self.tableWidget.verticalHeader().setStyleSheet(
            "QHeaderView::section{"
            "border-top:0.5px solid #E5E5E5;"
            "border-left:0px solid #E5E5E5;"
            "border-right:0.5px solid #E5E5E5;"
            "border-bottom: 0px solid #E5E5E5;"
            "background-color:white;"
            "padding:0.5px;"
            "}"
        )
        self.tableWidget.setHorizontalHeaderLabels(['Serial number', 'Time'])

        # 信号与槽
        self.pushButton_Init.clicked.connect(self.on_pushButton_Init_clicked)
        self.pushButton_CMD_CX.clicked.connect(self.on_pushButton_CMD_CX_clicked)
        self.pushButton_CMD_DeviceInfo.clicked.connect(self.on_pushButton_CMD_DeviceInfo_clicked)
        self.pushButton_User_clear.clicked.connect(self.pushButton_User_clear_clicked)
        self.pushButton_FeedBack_clear.clicked.connect(self.pushButton_FeedBack_clear_clicked)
        self.pushButton_CMD_SetParam.clicked.connect(self.on_pushButton_CMD_SetParam_clicked)
        self.pushButton_ChooseTwoPic.clicked.connect(self.on_pushButton_ChooseTwoPic_clicked)
        self.pushButton_ChooseEightPic.clicked.connect(self.on_pushButton_ChooseEightPic_clicked)
        self.pushButton_ChooseBin.clicked.connect(self.on_pushButton_ChooseBin_clicked)
        self.pushButton_SendTwo.clicked.connect(self.on_pushButton_SendTwo_clicked)
        self.pushButton_SendEight.clicked.connect(self.on_pushButton_SendEight_clicked)
        self.pushButton_SendBin.clicked.connect(self.on_pushButton_SendBin_clicked)
        self.pushButton_SendChar.clicked.connect(self.on_pushButton_SendChar_clicked)
        self.pushButton_CMD_SetSequence.clicked.connect(self.on_pushButton_CMD_SetSequence_clicked)
        self.pushButton_CMD_Play.clicked.connect(self.on_pushButton_CMD_Play_clicked)
        self.pushButton_CMD_PlayStop.clicked.connect(self.on_pushButton_CMD_PlayStop_clicked)
        self.pushButton_CMD_Stop.clicked.connect(self.on_pushButton_CMD_Stop_clicked)
        self.pushButton_CMD_UPDOWM.clicked.connect(self.on_pushButton_CMD_UPDOWM_clicked)
        self.pushButton_CMD_DateReverse.clicked.connect(self.on_pushButton_CMD_DateReverse_clicked)
        self.pushButton_CMD_SoftCF.clicked.connect(self.on_pushButton_CMD_SoftCF_clicked)
        self.pushButton_CMD_DMDFloat.clicked.connect(self.on_pushButton_CMD_DMDFloat_clicked)
        self.pushButton_CMD_Reset.clicked.connect(self.on_pushButton_CMD_Reset_clicked)
        self.pushButton_SendZDYCmd.clicked.connect(self.on_pushButton_SendZDYCmd_clicked)

    # 下位机反馈
    def FeedBack_TextEditShow(self, txt, sign):
        if sign == 0:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("black"))
            self.textEdit_FeedBack.mergeCurrentCharFormat(fmt)
            self.textEdit_FeedBack.insertPlainText(txt + "\n")
            self.feedback_cursor.movePosition(QTextCursor.End)
            self.textEdit_FeedBack.setTextCursor(self.feedback_cursor)
        elif sign == 1:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("red"))
            self.textEdit_FeedBack.mergeCurrentCharFormat(fmt)
            self.textEdit_FeedBack.insertPlainText(txt + "\n")
            self.feedback_cursor.movePosition(QTextCursor.End)
            self.textEdit_FeedBack.setTextCursor(self.feedback_cursor)

    # 用户操作与结果
    def User_TextEditShow(self, txt, sign):
        if sign == 0:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("black"))
            self.textEdit_User.mergeCurrentCharFormat(fmt)
            self.textEdit_User.insertPlainText(txt)
            self.user_cursor.movePosition(QTextCursor.End)
            self.textEdit_User.setTextCursor(self.user_cursor)
        elif sign == 1:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("red"))
            self.textEdit_User.mergeCurrentCharFormat(fmt)
            self.textEdit_User.insertPlainText(txt)
            self.user_cursor.movePosition(QTextCursor.End)
            self.textEdit_User.setTextCursor(self.user_cursor)

    # 初始化设备UDP
    def on_pushButton_Init_clicked(self):
        if self.pushButton_Init.text() == "Initialize Device UDP":
            print("Device UDP initialization started")
            # 获取设备ID
            self.Device_ID = int(self.comboBox_DeviceID.currentText())

            ZJ_IP = self.lineEdit_ZJ_IP.text().encode('utf-8')

            XWJ_IP = self.lineEdit_XWJ_IP.text().encode('utf-8')

            ZJ_Port = int(self.lineEdit_ZJ_Port.text())
            XWJ_Port = int(self.lineEdit_XWJ_Port.text())

            self.Device_ID = 1
            ZJ_IP_Init = "192.168.1.50"
            XWJ_IP_Init = "192.168.1.10"
            ZJ_IP = ZJ_IP_Init.encode('utf-8')

            XWJ_IP = XWJ_IP_Init.encode('utf-8')

            ZJ_Port = 1234
            XWJ_Port = 1234
            print("Device ID:", self.Device_ID)
            print("ZJ_IP:", ZJ_IP)
            print("XWJ_IP:", XWJ_IP)
            print("ZJ_Port:", ZJ_Port)
            print("XWJ_Port:", XWJ_Port)
            print("Current working directory:", os.getcwd())
            print("DLL loaded:", self.lib)
            try:
                res = self.lib.FL_DLP_HS_Init(self.Device_ID, ZJ_IP, ZJ_Port, XWJ_IP, XWJ_Port)
                print("FL_DLP_HS_Init result:", res)
            except Exception as e:
                print("Exception during DLL call:", e)

            print("Device ID:", self.Device_ID)
            print("ZJ_IP:", ZJ_IP)
            print("XWJ_IP:", XWJ_IP)
            print("ZJ_Port:", ZJ_Port)
            print("XWJ_Port:", XWJ_Port)
            self.User_TextEditShow("Device ID:" + str(self.Device_ID) + "\n", 0)
            self.User_TextEditShow("ZJ_IP:" + str(ZJ_IP, encoding='utf-8') + "\n", 0)
            self.User_TextEditShow("XWJ_IP:" + str(XWJ_IP, encoding='utf-8') + "\n", 0)
            self.User_TextEditShow("ZJ_Port:" + str(ZJ_Port) + "\n", 0)
            self.User_TextEditShow("XWJ_Port:" + str(XWJ_Port) + "\n", 0)

            if res != 0:
                self.User_TextEditShow("Device UDP initialization failed with failure code:" + str(res) + "\n", 1)

            else:
                self.User_TextEditShow("Device UDP initialization succeeded\n", 0)
                self.Thread_receive = udppthread()
                self.Thread_receive.FeedBack_TextEditShow.signal.connect(self.FeedBack_TextEditShow)

                self.Thread_receive.Device_ID = self.Device_ID
                self.Thread_receive.lib = self.lib
                self.Thread_receive.Rec_BS = True
                self.Thread_receive.start()

                self.comboBox_DeviceID.setDisabled(True)
                self.lineEdit_ZJ_IP.setDisabled(True)
                self.lineEdit_XWJ_IP.setDisabled(True)
                self.lineEdit_ZJ_Port.setDisabled(True)
                self.lineEdit_XWJ_Port.setDisabled(True)

                self.pushButton_Init.setText("Turn off device UDP")
        else:
            self.pushButton_Init.setText("Initialize Device UDP")
            self.Thread_receive.Rec_BS = False
            self.lib.FL_DLP_HS_DeInit(self.Device_ID)

            self.comboBox_DeviceID.setDisabled(False)
            self.lineEdit_ZJ_IP.setDisabled(False)
            self.lineEdit_XWJ_IP.setDisabled(False)
            self.lineEdit_ZJ_Port.setDisabled(False)
            self.lineEdit_XWJ_Port.setDisabled(False)
            self.User_TextEditShow("Device UDP closed successfully\n", 0)

    # 发送查询命令
    def on_pushButton_CMD_CX_clicked(self):
        self.Thread_receive.XC_BS = True
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, 7)
        if res != 0:
            self.Thread_receive.XC_BS = False
            self.User_TextEditShow("Failed to send query command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Succeeded in sending query command\n", 0)

    # 加载设备信息
    def on_pushButton_CMD_DeviceInfo_clicked(self):
        DeviceInfolen = c_int(0)
        DeviceInfo = (c_ubyte * 2048)()
        DeviceInfo2 = ''
        self.textEdit_DeviceInfo.clear()

        res = self.lib.FL_DLP_HS_AnalysisDevice(self.Device_ID, self.Thread_receive.XC_Data, DeviceInfo,
                                                byref(DeviceInfolen))
        if res != 0:
            self.User_TextEditShow("Failed to load device information, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Successfully loaded device information\n", 0)
            for i in range(DeviceInfolen.value):
                DeviceInfo2 += chr(DeviceInfo[i])

            DeviceList = DeviceInfo2.split(";")

            for i in range(len(DeviceList)):
                self.textEdit_DeviceInfo.insertPlainText(DeviceList[i] + "\n")

    # 清除用户操作
    def pushButton_User_clear_clicked(self):
        self.textEdit_User.clear()

    # 清除下位机消息
    def pushButton_FeedBack_clear_clicked(self):
        self.textEdit_FeedBack.clear()

    # 设置参数
    def on_pushButton_CMD_SetParam_clicked(self):
        Param_Delay = int(self.lineEdit_Param_Delay.text())

        # 反极性
        Param_Inverse = 0
        if self.checkBox_Param_SR_UP.isChecked() and self.checkBox_Param_SC_UP.isChecked():
            Param_Inverse = 0
        elif self.checkBox_Param_SR_UP.isChecked() and self.checkBox_Param_SC_DOWM.isChecked():
            Param_Inverse = 1
        elif self.checkBox_Param_SR_DOWM.isChecked() and self.checkBox_Param_SC_UP.isChecked():
            Param_Inverse = 2
        elif self.checkBox_Param_SR_DOWM.isChecked() and self.checkBox_Param_SC_DOWM.isChecked():
            Param_Inverse = 3

        # 灰度等级
        Param_Gray = int(self.lineEdit_Param_Gray.text())
        # 触发间隔
        Param_CFPicnum = int(self.lineEdit_Param_CFPicnum.text())
        # 触发延时
        Param_CFDelay = int(self.lineEdit_Param_CFDelay.text())

        res = self.lib.FL_DLP_HS_Send_Cmd_SetParam(self.Device_ID, Param_Delay, Param_Inverse, Param_Gray,
                                                   Param_CFPicnum, Param_CFDelay)
        if res != 0:
            self.User_TextEditShow("Failed to send setting parameters, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Sending setting parameters succeeded\n", 0)

    # 选择二值图像文件夹
    def on_pushButton_ChooseTwoPic_clicked(self):
        selectDir = QFileDialog.getExistingDirectory(self, "Select the binary image transmission folder", "C:/")
        self.Pic_AddrTwo = selectDir
        dir1 = QDir(selectDir)
        filter1 = ['*.bmp']
        fileInfo = dir1.entryInfoList(filter1, QDir.Files | QDir.Readable, QDir.Name)
        self.lineEdit_Picnum_Dir.setText(str(len(fileInfo)))

    # 选择灰度文件夹
    def on_pushButton_ChooseEightPic_clicked(self):
        selectDir = QFileDialog.getExistingDirectory(self, "Select the grayscale picture transfer folder", "C:/")
        self.Pic_AddrEight = selectDir
        dir1 = QDir(selectDir)
        filter1 = ['*.bmp']
        fileInfo = dir1.entryInfoList(filter1, QDir.Files | QDir.Readable, QDir.Name)
        self.lineEdit_Picnum_Dir.setText(str(len(fileInfo)))

    # 选择Bin文件
    def on_pushButton_ChooseBin_clicked(self):
        tmp = QFileDialog.getOpenFileName(self, "Select a single Bin file", "C:/", "BIN file(*.bin)")
        if tmp is not None:
            self.Pic_AddrBin = str(tmp[0])

    # 发送二值图像
    def on_pushButton_SendTwo_clicked(self):
        sendpic_type = 1
        pic_num = int(self.lineEdit_SendPicnum.text())
        Param_StartPicPosition = int(self.lineEdit_SendPicStarP.text())

        # 转化str to char* 方法一 转单个文件路径存在错误
        # Pic_AddrTwo1 = (c_ubyte * len(self.Pic_AddrTwo))(*bytearray(self.Pic_AddrTwo.encode()))
        # 转化str to char* 方法二
        Pic_AddrTwo1 = create_string_buffer(self.Pic_AddrTwo.encode())

        res = self.lib.FL_DLP_HS_Send_PICDATA(self.Device_ID, sendpic_type, Pic_AddrTwo1, pic_num,
                                              Param_StartPicPosition, None)
        if res != 0:
            self.User_TextEditShow("Failed to send binary image, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Sending binary image succeeded\n", 0)

    # 发送灰度文件夹
    def on_pushButton_SendEight_clicked(self):
        sendpic_type = 2
        pic_num = int(self.lineEdit_SendPicnum.text())
        Param_StartPicPosition = int(self.lineEdit_SendPicStarP.text())

        # 转化str to char* 方法二
        Pic_AddrEight1 = create_string_buffer(self.Pic_AddrEight.encode())

        res = self.lib.FL_DLP_HS_Send_PICDATA(self.Device_ID, sendpic_type, Pic_AddrEight1, pic_num,
                                              Param_StartPicPosition, None)
        if res != 0:
            self.User_TextEditShow("Failed to send grayscale image, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Successfully sent grayscale image\n", 0)

    # 发送Bin文件
    def on_pushButton_SendBin_clicked(self):
        sendpic_type = 3
        pic_num = int(self.lineEdit_SendPicnum.text())
        Param_StartPicPosition = int(self.lineEdit_SendPicStarP.text())
        # 转化str to char* 方法二
        Pic_AddrBin1 = create_string_buffer(self.Pic_AddrBin.encode())

        res = self.lib.FL_DLP_HS_Send_PICDATA(self.Device_ID, sendpic_type, Pic_AddrBin1, pic_num,
                                              Param_StartPicPosition, None)
        if res != 0:
            self.User_TextEditShow("Failed to send Bin file, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Bin file sent successfully\n", 0)

    # 二值图数组Demo
    def on_pushButton_SendChar_clicked(self):
        width = int(self.lineEdit_Width.text())
        height = int(self.lineEdit_Height.text())
        # Demo 为 8张数图片的数组：实际上数组内可以存储任意8的倍数图片数据
        picchar_s = ''

        for j in range(int(height / 4)):
            for k in range(int(width / 8)):
                picchar_s += '00'
        for j in range(int(height / 4)):
            for k in range(int(width / 8)):
                picchar_s += 'ff'
        for j in range(int(height / 4)):
            for k in range(int(width / 8)):
                picchar_s += '00'
        for j in range(int(height / 4)):
            for k in range(int(width / 8)):
                picchar_s += 'ff'

        picchar_b = bytes(bytearray.fromhex(picchar_s))
        picchar2 = cast(picchar_b, c_char_p)

        res = self.lib.FL_DLP_HS_Send_PICDATA(self.Device_ID, 4, None, 1, 1, picchar2)
        if res != 0:
            self.User_TextEditShow("Failed to send binary graph array Demo, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Successfully sent the binary graph array Demo\n", 0)

    # 设置序列
    def on_pushButton_CMD_SetSequence_clicked(self):
        Sequence_num = 0
        Sequence_Add = [int] * 8192
        Sequence_Time = [int] * 8192

        i = 0
        while self.tableWidget.item(i, 0) is not None and self.tableWidget.item(i, 0).text() is not None:
            Sequence_Add[i] = int(self.tableWidget.item(i, 0).text())

            if self.tableWidget.item(i, 1) is not None and self.tableWidget.item(i, 1).text() is not None:
                Sequence_Time[i] = int(self.tableWidget.item(i, 1).text())

            else:
                return

            i = i + 1

        Sequence_Add_byte = b''
        Sequence_Time_byte = b''
        for j in range(i):
            Sequence_Add_byte += (Sequence_Add[j]).to_bytes(length=4, byteorder='big')
            Sequence_Time_byte += (Sequence_Time[j]).to_bytes(length=4, byteorder='big')

        Sequence_num = i
        Sequence_Add2 = cast(Sequence_Add_byte, c_char_p)
        Sequence_Time2 = cast(Sequence_Time_byte, c_char_p)
        res = self.lib.FL_DLP_HS_Send_CMD_SetPlaySequence(self.Device_ID, Sequence_num, Sequence_Add2, Sequence_Time2)
        if res != 0:
            self.User_TextEditShow("Failed to send setting sequence, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Succeeded in sending setting sequence\n", 0)

    # 播放
    def on_pushButton_CMD_Play_clicked(self):
        type1 = 0
        if self.comboBox_CMD_Play.currentText() == "Internal single play":
            type1 = 1
        elif self.comboBox_CMD_Play.currentText() == "Internal loop play":
            type1 = 2
        elif self.comboBox_CMD_Play.currentText() == "External single play":
            type1 = 3
        elif self.comboBox_CMD_Play.currentText() == "External loop play":
            type1 = 4
        elif self.comboBox_CMD_Play.currentText() == "Variable sequence internal single play":
            type1 = 14
        elif self.comboBox_CMD_Play.currentText() == "Variable sequence internal loop play":
            type1 = 15
        elif self.comboBox_CMD_Play.currentText() == "Variable sequence external single play":
            type1 = 16
        elif self.comboBox_CMD_Play.currentText() == "Variable sequence external loop play":
            type1 = 17

        Param_StartP = int(self.lineEdit_PlayOffset.text())
        Param_PlayPicnum = int(self.lineEdit_PicCount.text())

        res = self.lib.FL_DLP_HS_Send_CMD_Play(self.Device_ID, type1, Param_StartP, Param_PlayPicnum)
        if res != 0:
            self.User_TextEditShow("Failed to send play command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Successfully sent the play command\n", 0)

    # 暂停
    def on_pushButton_CMD_PlayStop_clicked(self):
        type1 = 5
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, type1)
        if res != 0:
            self.lib.User_TextEditShow("Failed to send pause command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Successfully sent the pause command\n", 0)

    # 停止
    def on_pushButton_CMD_Stop_clicked(self):
        type1 = 8
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, type1)
        if res != 0:
            self.lib.User_TextEditShow("Failed to send stop command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Successfully sent the stop command\n", 0)

    # 上下颠倒
    def on_pushButton_CMD_UPDOWM_clicked(self):
        type1 = 11
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, type1)
        if res != 0:
            self.lib.User_TextEditShow("Failed to send the upside down command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Successfully sent the command of upside down\n", 0)

    # 数据反向
    def on_pushButton_CMD_DateReverse_clicked(self):
        type1 = 12
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, type1)
        if res != 0:
            self.lib.User_TextEditShow("Failed to send data reverse command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Sending data reverse command succeeded\n", 0)

    # 软件触发
    def on_pushButton_CMD_SoftCF_clicked(self):
        type1 = 13
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, type1)
        if res != 0:
            self.lib.User_TextEditShow("Failed to send software trigger command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("Succeeded in sending software trigger command\n", 0)

    # DMD-Float
    def on_pushButton_CMD_DMDFloat_clicked(self):
        type1 = 9
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, type1)
        if res != 0:
            self.lib.User_TextEditShow("Failed to send DMD Float command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("DMD Float command sent successfully\n", 0)

    # DMD-复位
    def on_pushButton_CMD_Reset_clicked(self):
        type1 = 10
        res = self.lib.FL_DLP_HS_Send_Fixed_Cmd_Noparam(self.Device_ID, type1)
        if res != 0:
            self.lib.User_TextEditShow("Failed to send DMD reset command, failure code:" + str(res) + "\n", 1)
        else:
            self.User_TextEditShow("DMD reset command sent successfully\n", 0)

    # 发送自定义命令
    def on_pushButton_SendZDYCmd_clicked(self):
        CMD = self.textEdit_ZDY.toPlainText()
        if CMD == '':
            self.User_TextEditShow("The input command is empty！\n", 1)
        else:
            sendBuf = bytes(bytearray.fromhex(CMD))
            CMDData = cast(sendBuf, c_char_p)
            res = self.lib.FL_DLP_HS_Send_Custom_Cmd(self.Device_ID, CMDData, len(sendBuf))
            if res != 0:
                self.User_TextEditShow("Failed to send custom command, failure code:" + str(res) + "\n", 1)
            else:
                self.User_TextEditShow("Successfully sent custom command\n", 0)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    b = QTabWidget()

    w0 = qtwindow()
    # w1 = qtwindow()
    # w2 = qtwindow()
    # w3 = qtwindow()
    # w4 = qtwindow()

    b.insertTab(0, w0, "Device 1")
    # b.insertTab(1, w1, "Device 2")
    # b.insertTab(2, w2, "Device 3")
    # b.insertTab(3, w3, "Device 4")
    # b.insertTab(4, w4, "Device 5")

    b.setMinimumSize(1000, 800)
    b.setWindowTitle("Python_FL_DLP_HS_Demo")
    b.show()
    b.setCurrentIndex(0)
    # b.setCurrentIndex(1)
    # b.setCurrentIndex(2)
    # b.setCurrentIndex(3)
    # b.setCurrentIndex(4)
    # b.setCurrentIndex(0)

    # 退出程序
    app.exec()
