# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1099, 847)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_16 = QGridLayout(self.centralwidget)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        font = QFont()
        font.setFamily(u"Cambria")
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.gridLayout_9 = QGridLayout(self.groupBox_3)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.gridLayout_8 = QGridLayout()
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.pushButton_FeedBack_clear = QPushButton(self.groupBox_3)
        self.pushButton_FeedBack_clear.setObjectName(u"pushButton_FeedBack_clear")
        self.pushButton_FeedBack_clear.setFont(font)

        self.gridLayout_8.addWidget(self.pushButton_FeedBack_clear, 0, 0, 1, 1)

        self.textEdit_FeedBack = QTextEdit(self.groupBox_3)
        self.textEdit_FeedBack.setObjectName(u"textEdit_FeedBack")
        self.textEdit_FeedBack.setFont(font)
        self.textEdit_FeedBack.setReadOnly(True)

        self.gridLayout_8.addWidget(self.textEdit_FeedBack, 1, 0, 1, 1)


        self.gridLayout_9.addLayout(self.gridLayout_8, 0, 0, 1, 1)


        self.gridLayout_7.addWidget(self.groupBox_3, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout_7, 4, 1, 2, 2)

        self.gridLayout_24 = QGridLayout()
        self.gridLayout_24.setObjectName(u"gridLayout_24")
        self.groupBox_9 = QGroupBox(self.centralwidget)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.groupBox_9.setFont(font)
        self.gridLayout_27 = QGridLayout(self.groupBox_9)
        self.gridLayout_27.setObjectName(u"gridLayout_27")
        self.gridLayout_25 = QGridLayout()
        self.gridLayout_25.setObjectName(u"gridLayout_25")
        self.textEdit_ZDY = QTextEdit(self.groupBox_9)
        self.textEdit_ZDY.setObjectName(u"textEdit_ZDY")
        self.textEdit_ZDY.setFont(font)

        self.gridLayout_25.addWidget(self.textEdit_ZDY, 0, 0, 1, 1)

        self.pushButton_SendZDYCmd = QPushButton(self.groupBox_9)
        self.pushButton_SendZDYCmd.setObjectName(u"pushButton_SendZDYCmd")
        self.pushButton_SendZDYCmd.setFont(font)

        self.gridLayout_25.addWidget(self.pushButton_SendZDYCmd, 1, 0, 1, 1)


        self.gridLayout_27.addLayout(self.gridLayout_25, 0, 0, 1, 1)


        self.gridLayout_24.addWidget(self.groupBox_9, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout_24, 5, 3, 1, 2)

        self.gridLayout_10 = QGridLayout()
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setFont(font)
        self.gridLayout_12 = QGridLayout(self.groupBox_4)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.gridLayout_11 = QGridLayout()
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.lineEdit_Param_CFDelay = QLineEdit(self.groupBox_4)
        self.lineEdit_Param_CFDelay.setObjectName(u"lineEdit_Param_CFDelay")
        self.lineEdit_Param_CFDelay.setFont(font)

        self.gridLayout_11.addWidget(self.lineEdit_Param_CFDelay, 5, 1, 1, 2)

        self.label_11 = QLabel(self.groupBox_4)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font)

        self.gridLayout_11.addWidget(self.label_11, 4, 0, 1, 1)

        self.checkBox_Param_SC_DOWM = QCheckBox(self.groupBox_4)
        self.buttonGroup_2 = QButtonGroup(MainWindow)
        self.buttonGroup_2.setObjectName(u"buttonGroup_2")
        self.buttonGroup_2.addButton(self.checkBox_Param_SC_DOWM)
        self.checkBox_Param_SC_DOWM.setObjectName(u"checkBox_Param_SC_DOWM")
        self.checkBox_Param_SC_DOWM.setFont(font)

        self.gridLayout_11.addWidget(self.checkBox_Param_SC_DOWM, 2, 2, 1, 1)

        self.checkBox_Param_SR_UP = QCheckBox(self.groupBox_4)
        self.buttonGroup = QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName(u"buttonGroup")
        self.buttonGroup.addButton(self.checkBox_Param_SR_UP)
        self.checkBox_Param_SR_UP.setObjectName(u"checkBox_Param_SR_UP")
        self.checkBox_Param_SR_UP.setFont(font)
        self.checkBox_Param_SR_UP.setChecked(True)

        self.gridLayout_11.addWidget(self.checkBox_Param_SR_UP, 1, 1, 1, 1)

        self.label_9 = QLabel(self.groupBox_4)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setFont(font)

        self.gridLayout_11.addWidget(self.label_9, 2, 0, 1, 1)

        self.label_10 = QLabel(self.groupBox_4)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setFont(font)

        self.gridLayout_11.addWidget(self.label_10, 3, 0, 1, 1)

        self.checkBox_Param_SC_UP = QCheckBox(self.groupBox_4)
        self.buttonGroup_2.addButton(self.checkBox_Param_SC_UP)
        self.checkBox_Param_SC_UP.setObjectName(u"checkBox_Param_SC_UP")
        self.checkBox_Param_SC_UP.setFont(font)
        self.checkBox_Param_SC_UP.setChecked(True)

        self.gridLayout_11.addWidget(self.checkBox_Param_SC_UP, 2, 1, 1, 1)

        self.lineEdit_Param_CFPicnum = QLineEdit(self.groupBox_4)
        self.lineEdit_Param_CFPicnum.setObjectName(u"lineEdit_Param_CFPicnum")
        self.lineEdit_Param_CFPicnum.setFont(font)

        self.gridLayout_11.addWidget(self.lineEdit_Param_CFPicnum, 4, 1, 1, 2)

        self.label_8 = QLabel(self.groupBox_4)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)

        self.gridLayout_11.addWidget(self.label_8, 1, 0, 1, 1)

        self.lineEdit_Param_Gray = QLineEdit(self.groupBox_4)
        self.lineEdit_Param_Gray.setObjectName(u"lineEdit_Param_Gray")
        self.lineEdit_Param_Gray.setFont(font)

        self.gridLayout_11.addWidget(self.lineEdit_Param_Gray, 3, 1, 1, 2)

        self.lineEdit_Param_Delay = QLineEdit(self.groupBox_4)
        self.lineEdit_Param_Delay.setObjectName(u"lineEdit_Param_Delay")
        self.lineEdit_Param_Delay.setFont(font)

        self.gridLayout_11.addWidget(self.lineEdit_Param_Delay, 0, 1, 1, 2)

        self.label_7 = QLabel(self.groupBox_4)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font)

        self.gridLayout_11.addWidget(self.label_7, 0, 0, 1, 1)

        self.label_12 = QLabel(self.groupBox_4)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font)

        self.gridLayout_11.addWidget(self.label_12, 5, 0, 1, 1)

        self.checkBox_Param_SR_DOWM = QCheckBox(self.groupBox_4)
        self.buttonGroup.addButton(self.checkBox_Param_SR_DOWM)
        self.checkBox_Param_SR_DOWM.setObjectName(u"checkBox_Param_SR_DOWM")
        self.checkBox_Param_SR_DOWM.setFont(font)

        self.gridLayout_11.addWidget(self.checkBox_Param_SR_DOWM, 1, 2, 1, 1)

        self.pushButton_CMD_SetParam = QPushButton(self.groupBox_4)
        self.pushButton_CMD_SetParam.setObjectName(u"pushButton_CMD_SetParam")
        self.pushButton_CMD_SetParam.setFont(font)

        self.gridLayout_11.addWidget(self.pushButton_CMD_SetParam, 6, 0, 1, 3)


        self.gridLayout_12.addLayout(self.gridLayout_11, 0, 0, 1, 1)


        self.gridLayout_10.addWidget(self.groupBox_4, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout_10, 0, 2, 1, 1)

        self.gridLayout_21 = QGridLayout()
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.groupBox_8 = QGroupBox(self.centralwidget)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setFont(font)
        self.gridLayout_23 = QGridLayout(self.groupBox_8)
        self.gridLayout_23.setObjectName(u"gridLayout_23")
        self.gridLayout_22 = QGridLayout()
        self.gridLayout_22.setObjectName(u"gridLayout_22")
        self.pushButton_CMD_UPDOWM = QPushButton(self.groupBox_8)
        self.pushButton_CMD_UPDOWM.setObjectName(u"pushButton_CMD_UPDOWM")
        self.pushButton_CMD_UPDOWM.setFont(font)

        self.gridLayout_22.addWidget(self.pushButton_CMD_UPDOWM, 0, 0, 1, 1)

        self.pushButton_CMD_DateReverse = QPushButton(self.groupBox_8)
        self.pushButton_CMD_DateReverse.setObjectName(u"pushButton_CMD_DateReverse")
        self.pushButton_CMD_DateReverse.setFont(font)

        self.gridLayout_22.addWidget(self.pushButton_CMD_DateReverse, 0, 1, 1, 1)

        self.pushButton_CMD_SoftCF = QPushButton(self.groupBox_8)
        self.pushButton_CMD_SoftCF.setObjectName(u"pushButton_CMD_SoftCF")
        self.pushButton_CMD_SoftCF.setFont(font)

        self.gridLayout_22.addWidget(self.pushButton_CMD_SoftCF, 0, 2, 1, 1)

        self.pushButton_CMD_DMDFloat = QPushButton(self.groupBox_8)
        self.pushButton_CMD_DMDFloat.setObjectName(u"pushButton_CMD_DMDFloat")
        self.pushButton_CMD_DMDFloat.setFont(font)

        self.gridLayout_22.addWidget(self.pushButton_CMD_DMDFloat, 1, 2, 1, 1)

        self.pushButton_CMD_Reset = QPushButton(self.groupBox_8)
        self.pushButton_CMD_Reset.setObjectName(u"pushButton_CMD_Reset")
        self.pushButton_CMD_Reset.setFont(font)

        self.gridLayout_22.addWidget(self.pushButton_CMD_Reset, 1, 0, 1, 1)


        self.gridLayout_23.addLayout(self.gridLayout_22, 0, 0, 1, 1)


        self.gridLayout_21.addWidget(self.groupBox_8, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout_21, 4, 4, 1, 1)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.gridLayout_6 = QGridLayout(self.groupBox_2)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.pushButton_User_clear = QPushButton(self.groupBox_2)
        self.pushButton_User_clear.setObjectName(u"pushButton_User_clear")
        self.pushButton_User_clear.setFont(font)

        self.gridLayout_5.addWidget(self.pushButton_User_clear, 0, 0, 1, 1)

        self.textEdit_User = QTextEdit(self.groupBox_2)
        self.textEdit_User.setObjectName(u"textEdit_User")
        self.textEdit_User.setFont(font)
        self.textEdit_User.setReadOnly(True)

        self.gridLayout_5.addWidget(self.textEdit_User, 1, 0, 1, 1)


        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)


        self.gridLayout_4.addWidget(self.groupBox_2, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout_4, 3, 0, 3, 1)

        self.gridLayout_19 = QGridLayout()
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.groupBox_7 = QGroupBox(self.centralwidget)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setFont(font)
        self.gridLayout_26 = QGridLayout(self.groupBox_7)
        self.gridLayout_26.setObjectName(u"gridLayout_26")
        self.gridLayout_20 = QGridLayout()
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.comboBox_CMD_Play = QComboBox(self.groupBox_7)
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.addItem("")
        self.comboBox_CMD_Play.setObjectName(u"comboBox_CMD_Play")
        self.comboBox_CMD_Play.setFont(font)

        self.gridLayout_20.addWidget(self.comboBox_CMD_Play, 1, 0, 1, 2)

        self.label_19 = QLabel(self.groupBox_7)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setFont(font)

        self.gridLayout_20.addWidget(self.label_19, 3, 0, 1, 1)

        self.label_17 = QLabel(self.groupBox_7)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setFont(font)

        self.gridLayout_20.addWidget(self.label_17, 2, 0, 1, 1)

        self.lineEdit_PlayOffset = QLineEdit(self.groupBox_7)
        self.lineEdit_PlayOffset.setObjectName(u"lineEdit_PlayOffset")
        self.lineEdit_PlayOffset.setFont(font)

        self.gridLayout_20.addWidget(self.lineEdit_PlayOffset, 2, 1, 1, 1)

        self.lineEdit_PicCount = QLineEdit(self.groupBox_7)
        self.lineEdit_PicCount.setObjectName(u"lineEdit_PicCount")
        self.lineEdit_PicCount.setFont(font)

        self.gridLayout_20.addWidget(self.lineEdit_PicCount, 3, 1, 1, 1)

        self.pushButton_CMD_PlayStop = QPushButton(self.groupBox_7)
        self.pushButton_CMD_PlayStop.setObjectName(u"pushButton_CMD_PlayStop")
        self.pushButton_CMD_PlayStop.setFont(font)

        self.gridLayout_20.addWidget(self.pushButton_CMD_PlayStop, 5, 0, 1, 1)

        self.pushButton_CMD_Stop = QPushButton(self.groupBox_7)
        self.pushButton_CMD_Stop.setObjectName(u"pushButton_CMD_Stop")
        self.pushButton_CMD_Stop.setFont(font)

        self.gridLayout_20.addWidget(self.pushButton_CMD_Stop, 5, 1, 1, 1)

        self.pushButton_CMD_Play = QPushButton(self.groupBox_7)
        self.pushButton_CMD_Play.setObjectName(u"pushButton_CMD_Play")
        self.pushButton_CMD_Play.setFont(font)

        self.gridLayout_20.addWidget(self.pushButton_CMD_Play, 4, 0, 1, 2)


        self.gridLayout_26.addLayout(self.gridLayout_20, 0, 0, 1, 1)


        self.gridLayout_19.addWidget(self.groupBox_7, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout_19, 2, 4, 2, 1)

        self.gridLayout_13 = QGridLayout()
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.groupBox_5 = QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setFont(font)
        self.gridLayout_14 = QGridLayout(self.groupBox_5)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.gridLayout_15 = QGridLayout()
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.pushButton_ChooseBin = QPushButton(self.groupBox_5)
        self.pushButton_ChooseBin.setObjectName(u"pushButton_ChooseBin")
        self.pushButton_ChooseBin.setFont(font)

        self.gridLayout_15.addWidget(self.pushButton_ChooseBin, 0, 2, 1, 1)

        self.pushButton_SendTwo = QPushButton(self.groupBox_5)
        self.pushButton_SendTwo.setObjectName(u"pushButton_SendTwo")
        self.pushButton_SendTwo.setFont(font)

        self.gridLayout_15.addWidget(self.pushButton_SendTwo, 4, 0, 1, 1)

        self.lineEdit_SendPicnum = QLineEdit(self.groupBox_5)
        self.lineEdit_SendPicnum.setObjectName(u"lineEdit_SendPicnum")
        self.lineEdit_SendPicnum.setFont(font)

        self.gridLayout_15.addWidget(self.lineEdit_SendPicnum, 2, 1, 1, 2)

        self.label_13 = QLabel(self.groupBox_5)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setFont(font)

        self.gridLayout_15.addWidget(self.label_13, 3, 0, 1, 1)

        self.lineEdit_SendPicStarP = QLineEdit(self.groupBox_5)
        self.lineEdit_SendPicStarP.setObjectName(u"lineEdit_SendPicStarP")
        self.lineEdit_SendPicStarP.setFont(font)

        self.gridLayout_15.addWidget(self.lineEdit_SendPicStarP, 3, 1, 1, 2)

        self.pushButton_SendEight = QPushButton(self.groupBox_5)
        self.pushButton_SendEight.setObjectName(u"pushButton_SendEight")
        self.pushButton_SendEight.setFont(font)

        self.gridLayout_15.addWidget(self.pushButton_SendEight, 4, 1, 1, 1)

        self.label_14 = QLabel(self.groupBox_5)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setFont(font)

        self.gridLayout_15.addWidget(self.label_14, 1, 0, 1, 1)

        self.lineEdit_Picnum_Dir = QLineEdit(self.groupBox_5)
        self.lineEdit_Picnum_Dir.setObjectName(u"lineEdit_Picnum_Dir")
        self.lineEdit_Picnum_Dir.setFont(font)
        self.lineEdit_Picnum_Dir.setReadOnly(True)

        self.gridLayout_15.addWidget(self.lineEdit_Picnum_Dir, 1, 1, 1, 2)

        self.pushButton_ChooseEightPic = QPushButton(self.groupBox_5)
        self.pushButton_ChooseEightPic.setObjectName(u"pushButton_ChooseEightPic")
        self.pushButton_ChooseEightPic.setFont(font)

        self.gridLayout_15.addWidget(self.pushButton_ChooseEightPic, 0, 1, 1, 1)

        self.pushButton_ChooseTwoPic = QPushButton(self.groupBox_5)
        self.pushButton_ChooseTwoPic.setObjectName(u"pushButton_ChooseTwoPic")
        self.pushButton_ChooseTwoPic.setFont(font)

        self.gridLayout_15.addWidget(self.pushButton_ChooseTwoPic, 0, 0, 1, 1)

        self.label_15 = QLabel(self.groupBox_5)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setFont(font)

        self.gridLayout_15.addWidget(self.label_15, 2, 0, 1, 1)

        self.label_16 = QLabel(self.groupBox_5)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setFont(font)

        self.gridLayout_15.addWidget(self.label_16, 6, 0, 1, 1)

        self.pushButton_SendBin = QPushButton(self.groupBox_5)
        self.pushButton_SendBin.setObjectName(u"pushButton_SendBin")
        self.pushButton_SendBin.setFont(font)

        self.gridLayout_15.addWidget(self.pushButton_SendBin, 4, 2, 1, 1)

        self.pushButton_SendChar = QPushButton(self.groupBox_5)
        self.pushButton_SendChar.setObjectName(u"pushButton_SendChar")
        self.pushButton_SendChar.setFont(font)

        self.gridLayout_15.addWidget(self.pushButton_SendChar, 6, 1, 1, 2)

        self.label_18 = QLabel(self.groupBox_5)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setFont(font)

        self.gridLayout_15.addWidget(self.label_18, 5, 0, 1, 1)

        self.lineEdit_Width = QLineEdit(self.groupBox_5)
        self.lineEdit_Width.setObjectName(u"lineEdit_Width")
        self.lineEdit_Width.setFont(font)

        self.gridLayout_15.addWidget(self.lineEdit_Width, 5, 1, 1, 1)

        self.lineEdit_Height = QLineEdit(self.groupBox_5)
        self.lineEdit_Height.setObjectName(u"lineEdit_Height")
        self.lineEdit_Height.setFont(font)

        self.gridLayout_15.addWidget(self.lineEdit_Height, 5, 2, 1, 1)


        self.gridLayout_14.addLayout(self.gridLayout_15, 0, 0, 1, 1)


        self.gridLayout_13.addWidget(self.groupBox_5, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout_13, 1, 2, 3, 1)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setFont(font)
        self.gridLayout_3 = QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.lineEdit_XWJ_Port = QLineEdit(self.groupBox)
        self.lineEdit_XWJ_Port.setObjectName(u"lineEdit_XWJ_Port")
        self.lineEdit_XWJ_Port.setFont(font)

        self.gridLayout_2.addWidget(self.lineEdit_XWJ_Port, 2, 3, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)

        self.gridLayout_2.addWidget(self.label_3, 1, 2, 1, 1)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font)
        self.label_5.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 2)

        self.pushButton_Init = QPushButton(self.groupBox)
        self.pushButton_Init.setObjectName(u"pushButton_Init")
        self.pushButton_Init.setFont(font)

        self.gridLayout_2.addWidget(self.pushButton_Init, 3, 0, 1, 4)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setFont(font)

        self.gridLayout_2.addWidget(self.label_2, 2, 0, 1, 1)

        self.lineEdit_XWJ_IP = QLineEdit(self.groupBox)
        self.lineEdit_XWJ_IP.setObjectName(u"lineEdit_XWJ_IP")
        self.lineEdit_XWJ_IP.setFont(font)

        self.gridLayout_2.addWidget(self.lineEdit_XWJ_IP, 2, 1, 1, 1)

        self.lineEdit_ZJ_Port = QLineEdit(self.groupBox)
        self.lineEdit_ZJ_Port.setObjectName(u"lineEdit_ZJ_Port")
        self.lineEdit_ZJ_Port.setFont(font)

        self.gridLayout_2.addWidget(self.lineEdit_ZJ_Port, 1, 3, 1, 1)

        self.pushButton_CMD_CX = QPushButton(self.groupBox)
        self.pushButton_CMD_CX.setObjectName(u"pushButton_CMD_CX")
        self.pushButton_CMD_CX.setFont(font)

        self.gridLayout_2.addWidget(self.pushButton_CMD_CX, 4, 0, 1, 4)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setFont(font)

        self.gridLayout_2.addWidget(self.label, 1, 0, 1, 1)

        self.pushButton_CMD_DeviceInfo = QPushButton(self.groupBox)
        self.pushButton_CMD_DeviceInfo.setObjectName(u"pushButton_CMD_DeviceInfo")
        self.pushButton_CMD_DeviceInfo.setFont(font)

        self.gridLayout_2.addWidget(self.pushButton_CMD_DeviceInfo, 5, 0, 1, 4)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setFont(font)

        self.gridLayout_2.addWidget(self.label_4, 2, 2, 1, 1)

        self.lineEdit_ZJ_IP = QLineEdit(self.groupBox)
        self.lineEdit_ZJ_IP.setObjectName(u"lineEdit_ZJ_IP")
        self.lineEdit_ZJ_IP.setFont(font)

        self.gridLayout_2.addWidget(self.lineEdit_ZJ_IP, 1, 1, 1, 1)

        self.comboBox_DeviceID = QComboBox(self.groupBox)
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.addItem("")
        self.comboBox_DeviceID.setObjectName(u"comboBox_DeviceID")
        self.comboBox_DeviceID.setFont(font)

        self.gridLayout_2.addWidget(self.comboBox_DeviceID, 0, 2, 1, 2)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font)

        self.gridLayout_2.addWidget(self.label_6, 6, 0, 1, 1)

        self.textEdit_DeviceInfo = QTextEdit(self.groupBox)
        self.textEdit_DeviceInfo.setObjectName(u"textEdit_DeviceInfo")
        self.textEdit_DeviceInfo.setFont(font)
        self.textEdit_DeviceInfo.setReadOnly(True)

        self.gridLayout_2.addWidget(self.textEdit_DeviceInfo, 6, 1, 1, 3)


        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)


        self.gridLayout_16.addLayout(self.gridLayout, 0, 0, 3, 2)

        self.groupBox_6 = QGroupBox(self.centralwidget)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setFont(font)
        self.gridLayout_18 = QGridLayout(self.groupBox_6)
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.gridLayout_17 = QGridLayout()
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.tableWidget = QTableWidget(self.groupBox_6)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setFont(font)

        self.gridLayout_17.addWidget(self.tableWidget, 2, 0, 1, 1)

        self.label_20 = QLabel(self.groupBox_6)
        self.label_20.setObjectName(u"label_20")

        self.gridLayout_17.addWidget(self.label_20, 0, 0, 1, 1)

        self.pushButton_CMD_SetSequence = QPushButton(self.groupBox_6)
        self.pushButton_CMD_SetSequence.setObjectName(u"pushButton_CMD_SetSequence")
        self.pushButton_CMD_SetSequence.setFont(font)

        self.gridLayout_17.addWidget(self.pushButton_CMD_SetSequence, 3, 0, 1, 1)

        self.label_21 = QLabel(self.groupBox_6)
        self.label_21.setObjectName(u"label_21")

        self.gridLayout_17.addWidget(self.label_21, 1, 0, 1, 1)


        self.gridLayout_18.addLayout(self.gridLayout_17, 0, 0, 1, 1)


        self.gridLayout_16.addWidget(self.groupBox_6, 0, 4, 2, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1099, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Equipment feedback information", None))
        self.pushButton_FeedBack_clear.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("MainWindow", u"Custom Commands", None))
        self.pushButton_SendZDYCmd.setText(QCoreApplication.translate("MainWindow", u"Send custom commands", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Set parameters", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Trigger interval", None))
        self.checkBox_Param_SC_DOWM.setText(QCoreApplication.translate("MainWindow", u"Falling edge", None))
        self.checkBox_Param_SR_UP.setText(QCoreApplication.translate("MainWindow", u"Rising edge", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Output trigger mode", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Image gray scale\uff081-16\uff09", None))
        self.checkBox_Param_SC_UP.setText(QCoreApplication.translate("MainWindow", u"Rising edge", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Input trigger mode", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Picture playing frequency (Hz)", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Trigger delay", None))
        self.checkBox_Param_SR_DOWM.setText(QCoreApplication.translate("MainWindow", u"Falling edge", None))
        self.pushButton_CMD_SetParam.setText(QCoreApplication.translate("MainWindow", u"Send parameter setting command", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("MainWindow", u"\u5176\u4ed6\u64cd\u4f5c", None))
        self.pushButton_CMD_UPDOWM.setText(QCoreApplication.translate("MainWindow", u"upside down", None))
        self.pushButton_CMD_DateReverse.setText(QCoreApplication.translate("MainWindow", u"Data Reversal", None))
        self.pushButton_CMD_SoftCF.setText(QCoreApplication.translate("MainWindow", u"Software trigger", None))
        self.pushButton_CMD_DMDFloat.setText(QCoreApplication.translate("MainWindow", u"DMD-Float", None))
        self.pushButton_CMD_Reset.setText(QCoreApplication.translate("MainWindow", u"DMD-Reset", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"User actions and operation results", None))
        self.pushButton_User_clear.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("MainWindow", u"Play operation", None))
        self.comboBox_CMD_Play.setItemText(0, QCoreApplication.translate("MainWindow", u"Internal single play", None))
        self.comboBox_CMD_Play.setItemText(1, QCoreApplication.translate("MainWindow", u"Internal loop play", None))
        self.comboBox_CMD_Play.setItemText(2, QCoreApplication.translate("MainWindow", u"External single play", None))
        self.comboBox_CMD_Play.setItemText(3, QCoreApplication.translate("MainWindow", u"External loop play", None))
        self.comboBox_CMD_Play.setItemText(4, QCoreApplication.translate("MainWindow", u"Variable sequence internal single play", None))
        self.comboBox_CMD_Play.setItemText(5, QCoreApplication.translate("MainWindow", u"Variable sequence internal loop play", None))
        self.comboBox_CMD_Play.setItemText(6, QCoreApplication.translate("MainWindow", u"Variable sequence external single play", None))
        self.comboBox_CMD_Play.setItemText(7, QCoreApplication.translate("MainWindow", u"Variable sequence external loop play", None))

        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Number of pic", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Start position", None))
        self.pushButton_CMD_PlayStop.setText(QCoreApplication.translate("MainWindow", u"Pause", None))
        self.pushButton_CMD_Stop.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.pushButton_CMD_Play.setText(QCoreApplication.translate("MainWindow", u"Play", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Caching image function", None))
        self.pushButton_ChooseBin.setText(QCoreApplication.translate("MainWindow", u"Bin File", None))
        self.pushButton_SendTwo.setText(QCoreApplication.translate("MainWindow", u"Send Binary", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Start Cache Location", None))
        self.pushButton_SendEight.setText(QCoreApplication.translate("MainWindow", u"Send grayscale", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Number of pictures in the folder:", None))
        self.pushButton_ChooseEightPic.setText(QCoreApplication.translate("MainWindow", u"8 bit grayscale folder", None))
        self.pushButton_ChooseTwoPic.setText(QCoreApplication.translate("MainWindow", u"Binary image folder", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Number of pictures sent", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"image data array Demo", None))
        self.pushButton_SendBin.setText(QCoreApplication.translate("MainWindow", u"Send Bin", None))
        self.pushButton_SendChar.setText(QCoreApplication.translate("MainWindow", u"Send image data array", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Picture width and height", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"UDP connection and device initialization", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Host Port:", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Device ID:", None))
        self.pushButton_Init.setText(QCoreApplication.translate("MainWindow", u"Initialize Device UDP", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Device IP:", None))
        self.pushButton_CMD_CX.setText(QCoreApplication.translate("MainWindow", u"Send query command", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Host IP:", None))
        self.pushButton_CMD_DeviceInfo.setText(QCoreApplication.translate("MainWindow", u"Load device information", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Device Port:", None))
        self.comboBox_DeviceID.setItemText(0, QCoreApplication.translate("MainWindow", u"1", None))
        self.comboBox_DeviceID.setItemText(1, QCoreApplication.translate("MainWindow", u"2", None))
        self.comboBox_DeviceID.setItemText(2, QCoreApplication.translate("MainWindow", u"3", None))
        self.comboBox_DeviceID.setItemText(3, QCoreApplication.translate("MainWindow", u"4", None))
        self.comboBox_DeviceID.setItemText(4, QCoreApplication.translate("MainWindow", u"5", None))
        self.comboBox_DeviceID.setItemText(5, QCoreApplication.translate("MainWindow", u"6", None))
        self.comboBox_DeviceID.setItemText(6, QCoreApplication.translate("MainWindow", u"7", None))
        self.comboBox_DeviceID.setItemText(7, QCoreApplication.translate("MainWindow", u"8", None))
        self.comboBox_DeviceID.setItemText(8, QCoreApplication.translate("MainWindow", u"9", None))
        self.comboBox_DeviceID.setItemText(9, QCoreApplication.translate("MainWindow", u"10", None))

        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Device infor:", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"Set Sequence", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Serial number\uff1aStart From 0", None))
        self.pushButton_CMD_SetSequence.setText(QCoreApplication.translate("MainWindow", u"Set Sequence", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Time:unit: 5 nanoseconds; input>=1000", None))
    # retranslateUi

