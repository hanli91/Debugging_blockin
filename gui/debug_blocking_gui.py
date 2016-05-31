from functools import partial
from PyQt4 import QtGui, QtCore
from enum import Enum
from ranking_aggregation.rank_aggregation import DebuggingBlockingStat
import magellan as mg

import copy


class VerifButtonType(Enum):
    true = 1
    false = 2


class RetButtonType(Enum):
    unclear = 1
    terminate = 2


class MainWindowManager(QtGui.QWidget):

    def __init__(self, schema, recom_lists):
        super(MainWindowManager, self).__init__()
        self.schema = schema
        self.recom_lists = recom_lists
        self.debugging_stat = DebuggingBlockingStat()

        self.basic_info_view_widget = None
        self.tuple_pair_view_widget = None
        self.recom_list_view_widget = None
        self.setup_gui(recom_lists[0])
        width = min((40 + 1)*105, mg._viewapp.desktop().screenGeometry().width() - 50)
        height = min((50 + 1)*41, mg._viewapp.desktop().screenGeometry().width() - 100)
        self.resize(width, height)

    def setup_gui(self, recom_list):
        self.setWindowTitle('Blocking Debugger')
        self.recom_list_view = RecomListTableViewWithLabel(self, recom_list, 'Candidate List')
        self.tuple_pair_view_widget = TuplePairTableViewWithLabel(self, recom_list, self.schema, 'Tuple Pair')
        self.basic_info_view_widget = BasicInfoViewWithLabel(self, 'Basic Info')

        layout = QtGui.QGridLayout(self)
        horizonal_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        horizonal_splitter.addWidget(self.tuple_pair_view_widget)
        horizonal_splitter.addWidget(self.recom_list_view)
        horizonal_splitter.setStretchFactor(0, 2)
        horizonal_splitter.setStretchFactor(1, 1)
        #vertical_splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        #vertical_splitter.addWidget(self.basic_info_view_widget)
        #vertical_splitter.addWidget(horizonal_splitter)
        #vertical_splitter.setStretchFactor(0, 1)
        #vertical_splitter.setStretchFactor(1, 10)
        #layout.addWidget(vertical_splitter)
        layout.addWidget(self.basic_info_view_widget, 0, 0, 1, 1)
        layout.addWidget(horizonal_splitter, 1, 0, 10, 1)
        self.setLayout(layout)

    def handle_expand_button(self, index):
        self.tuple_pair_view_widget.update(index)

    def handle_verif_button(self, button, type, pair_id):
        total_pos_set = self.debugging_stat.total_pos_set
        total_neg_set = self.debugging_stat.total_neg_set
        cur_iter_pos_set = self.debugging_stat.cur_iter_pos_set
        if button.isChecked():
            if type == VerifButtonType.true:
                total_pos_set.add(pair_id)
                cur_iter_pos_set.add(pair_id)
                self.debugging_stat.cur_pos_num += 1
                self.debugging_stat.total_pos_num += 1
            if type == VerifButtonType.false:
                total_neg_set.add(pair_id)
                self.debugging_stat.cur_neg_num += 1
                self.debugging_stat.total_neg_num += 1
        else:
            if type == VerifButtonType.true:
                total_pos_set.remove(pair_id)
                cur_iter_pos_set.remove(pair_id)
                self.debugging_stat.cur_pos_num -= 1
                self.debugging_stat.total_pos_num -= 1
            if type == VerifButtonType.false:
                total_neg_set.remove(pair_id)
                self.debugging_stat.cur_neg_num -= 1
                self.debugging_stat.total_neg_num -= 1

        self.basic_info_view_widget.update_cur_iter_verified_num(
            type, self.debugging_stat.cur_pos_num, self.debugging_stat.cur_neg_num)
        self.basic_info_view_widget.update_total_verified_num(
            type, self.debugging_stat.total_pos_num, self.debugging_stat.total_neg_num)


class BasicInfoViewWithLabel(QtGui.QWidget):
    def __init__(self, controller, label):
        super(BasicInfoViewWithLabel, self).__init__()
        self.controller = controller
        self.groupbox = QtGui.QGroupBox(label)
        #self.name_label_widget = QtGui.QLabel(label)

        self.cur_iter_label_obj = None
        self.cur_pos_label_obj = None
        self.cur_neg_label_obj = None

        self.total_iter_label_obj = None
        self.total_pos_label_obj = None
        self.total_neg_label_obj = None

        self.ret_label_obj = None

        self.setup_gui()

    def setup_gui(self):
        layout = QtGui.QGridLayout()
        self.build_cur_iter_widget(layout)
        self.build_total_iter_widget(layout)
        self.build_ret_widget(layout)

        self.groupbox.setLayout(layout)
        wrapper_layout = QtGui.QHBoxLayout()
        wrapper_layout.addWidget(self.groupbox)
        self.setLayout(wrapper_layout)

    def build_cur_iter_widget(self, layout):
        self.cur_iter_label_obj = QtGui.QLabel('Current iteration: ' + str(1))
        self.cur_pos_label_obj = QtGui.QLabel('Current iteration verified true matching: ' + str(0))
        self.cur_neg_label_obj = QtGui.QLabel('Current iteration verified false matching: ' + str(0))
        layout.addWidget(self.cur_iter_label_obj, 0, 0)
        layout.addWidget(self.cur_pos_label_obj, 1, 0)
        layout.addWidget(self.cur_neg_label_obj, 2, 0)

    def update_cur_iter_verified_num(self, type, cur_pos_num, cur_neg_num):
        if type == VerifButtonType.true:
            self.cur_pos_label_obj.setText('Current iteration verified true matching: ' + str(cur_pos_num))
        if type == VerifButtonType.false:
            self.cur_neg_label_obj.setText('Current iteration verified false matching: ' + str(cur_neg_num))

    def build_total_iter_widget(self, layout):
        self.total_iter_label_obj = QtGui.QLabel('Total iterations: ' + str(1))
        self.total_pos_label_obj = QtGui.QLabel('Total verified true matching: ' + str(0))
        self.total_neg_label_obj = QtGui.QLabel('Total verified false matching: ' + str(0))
        layout.addWidget(self.total_iter_label_obj, 0, 1)
        layout.addWidget(self.total_pos_label_obj, 1, 1)
        layout.addWidget(self.total_neg_label_obj, 2, 1)

    def update_total_verified_num(self, type, total_pos_num, total_neg_num):
        if type == VerifButtonType.true:
            self.total_pos_label_obj.setText('Total verified true matching: ' + str(total_pos_num))
        if type == VerifButtonType.false:
            self.total_neg_label_obj.setText('Total verified false matching: ' + str(total_neg_num))

    def build_ret_widget(self, layout):
        self.ret_label_obj = QtGui.QLabel('I\'ve finished verifying the current iteration:')
        layout.addWidget(self.ret_label_obj, 0, 2)

        unclear_button = QtGui.QRadioButton('Give me a set of new tuple pairs')
        terminate_button = QtGui.QRadioButton('Terminate debugging blocking')
        finish_button = QtGui.QPushButton('Confirm', self)
        layout.addWidget(unclear_button, 1, 2)
        layout.addWidget(terminate_button, 1, 3)
        layout.addWidget(finish_button, 2, 2)


class RecomListTableViewWithLabel(QtGui.QWidget):

    def __init__(self, controller, recom_list, label):
        super(RecomListTableViewWithLabel, self).__init__()
        self.controller = controller
        self.recom_list = recom_list
        self.label = label
        self.label_widget = None
        self.table_view_widget = None
        self.setup_gui()

    def setup_gui(self):
        label = QtGui.QLabel(self.label)
        table_view = RecomListTableView(self.controller, self.recom_list)
        self.label_widget = label
        self.table_view_widget = table_view
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.label_widget)
        layout.addWidget(self.table_view_widget)
        self.setLayout(layout)


class RecomListTableView(QtGui.QTableWidget):

    def __init__(self, controller, recom_list):
        super(RecomListTableView, self).__init__()
        self.controller = controller
        self.recom_list = recom_list
        #self.checkbox_list = []
        self.setup_gui()

    def setup_gui(self):
        nrows = len(self.recom_list)
        self.setRowCount(nrows)
        ncols = 4
        self.setColumnCount(ncols)

        headers = ['Left Tuple ID', 'Right Tuple ID', 'Expand', 'Verification Result']
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setStretchLastSection(True)
        self.resizeRowsToContents()
        self.verticalHeader().setVisible(True)

        if nrows > 0:
            for i in range(nrows):
                #checkbox_pair = [QtGui.QRadioButton('True Matching'), QtGui.QRadioButton('False Matching')]
                for j in range(ncols):
                    if j < 2:
                        self.setItem(i, j, QtGui.QTableWidgetItem(str(self.recom_list[i][j])))
                        self.item(i, j).setFlags(QtCore.Qt.ItemIsEnabled)
                    if j == 2:
                        button = QtGui.QPushButton('Expand', self)
                        self.setCellWidget(i, j, button)
                        button.clicked.connect(partial(self.controller.handle_expand_button, i))
                    if j == 3:
                        layout = QtGui.QVBoxLayout()
                        true_button = QtGui.QRadioButton('True Matching')
                        false_button = QtGui.QRadioButton('False Matching')
                        true_button.toggled.connect(partial(self.controller.handle_verif_button, true_button,
                                                            VerifButtonType.true, (self.recom_list[i][0], self.recom_list[i][1])))
                        false_button.toggled.connect(partial(self.controller.handle_verif_button, false_button,
                                                             VerifButtonType.false, (self.recom_list[i][0], self.recom_list[i][1])))
                        layout.addWidget(true_button)
                        layout.addWidget(false_button)
                        #checkbox_pair[0].toggled.connect(partial(self.controller.handle_verif_button, i, 0))
                        #checkbox_pair[1].toggled.connect(partial(self.controller.handle_verif_button, i, 1))
                        #self.checkbox_list.append(checkbox_pair)
                        #layout.addWidget(checkbox_pair[0])
                        #layout.addWidget(checkbox_pair[1])
                        cellWidget = QtGui.QWidget()
                        cellWidget.setLayout(layout)

                        self.setCellWidget(i, j, cellWidget)

        self.resizeRowsToContents()
        self.resizeColumnsToContents()


class TuplePairTableViewWithLabel(QtGui.QWidget):

    def __init__(self, controller, recom_list, schema, label):
        super(TuplePairTableViewWithLabel, self).__init__()
        self.controller = controller
        self.recom_list = recom_list
        self.schema = schema
        self.label = label

        self.label_widget = None
        self.table_view_widget = None

        self.setup_gui()

    def setup_gui(self):
        label = QtGui.QLabel(self.label)
        table_view = TuplePairTableView(self.controller, self.schema)
        self.label_widget = label
        self.table_view_widget = table_view
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.label_widget)
        layout.addWidget(self.table_view_widget)
        self.setLayout(layout)

    def update(self, index):
        tuple_view = self.table_view_widget
        nrows = tuple_view.rowCount()
        ltuple = self.recom_list[index][2]
        rtuple = self.recom_list[index][3]
        for i in range(nrows):
            tuple_view.setItem(i, 0, QtGui.QTableWidgetItem(str(ltuple[i])))
            tuple_view.item(i, 0).setFlags(QtCore.Qt.ItemIsEnabled)
            tuple_view.setItem(i, 1, QtGui.QTableWidgetItem(str(rtuple[i])))
            tuple_view.item(i, 1).setFlags(QtCore.Qt.ItemIsEnabled)

        tuple_view.resizeRowsToContents()


class TuplePairTableView(QtGui.QTableWidget):

    def __init__(self, controller, schema):
        super(TuplePairTableView, self).__init__()
        self.controller = controller
        self.schema = schema

        self.setup_gui()

    def setup_gui(self):
        nrows = len(self.schema)
        self.setRowCount(nrows)
        ncols = 2
        self.setColumnCount(ncols)

        headers = ['Left Tuple Value', 'Right Tuple Value']
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(True)
        self.setVerticalHeaderLabels(self.schema)
        self.horizontalHeader().setResizeMode(1)
        # 0 represents "Disable selection"
        self.setSelectionMode(0)


def read_wrapped_recom_list(file, K):
    recom_lists = []
    infile = open(file, 'r')
    line = infile.readline()
    schema = line.split('@_@_@_@')
    for i in range(len(schema)):
        schema[i] = schema[i].rstrip('\n')
    lines = infile.readlines()
    start = 0
    while (start < len(lines)):
        recom_list = []
        for i in range(K):
            ltuple = []
            rtuple = []
            for j in range(len(schema)):
                ltuple.append(lines[start + 2 + j].rstrip('\n'))
                rtuple.append(lines[start + 2 + len(schema) + j].rstrip('\n'))
            recom_list.append((lines[start].rstrip('\n'), lines[start + 1].rstrip('\n'), ltuple, rtuple))
            start += 2 + 2 * len(schema)
        recom_lists.append(copy.deepcopy(recom_list))
    return schema, recom_lists


if __name__ == "__main__":
    schema, recom_lists = read_wrapped_recom_list('../Misc/new_recom_list.txt', 100)
    app = mg._viewapp
    window = MainWindowManager(schema, recom_lists)
    window.show()
    app.exec_()