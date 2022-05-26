import logging 
import time
import datetime

def make_list_data(dataset: dict):
    data_list = []
    for ex, feature_dict in dataset.items():
        feature_dict.update({'__id__': ex})
        data_list.append(feature_dict)
    return data_list

class MPLogger:
    def __init__(
            self,
            name,
            log_level,
            id=None,
            loop_report_time=None,
            ):
        if id is None:
            self.name = name
        else:
            self.name = f'{name}_{id}'
        self.id = id
        self.log_level = log_level
        self.loop_report_time = loop_report_time

    def initialize_loop_reporting(self, report_level=None):
        self.last_report_time = time.time()
        self.last_iteration_num = 0
        if report_level is None:
            report_level = logging.INFO
        self.report_level = report_level

    def report(self, message=None, i=None, total=None):
        if self.loop_report_time is None:
            return
        time_since_last_report = time.time() - self.last_report_time
        if time_since_last_report >= self.loop_report_time:
            if message is not None:
                if self.report_level == logging.DEBUG:
                    self.debug(message)
                elif self.report_level == logging.INFO:
                    self.info(message)
                elif self.report_level == logging.WARNING:
                    self.warning(message)
                elif self.report_level == logging.ERROR:
                    self.error(message)
                elif self.report_level == logging.CRITICAL:
                    self.critical(message)
            elif i is not None and total is not None:
                sec_per_iter = time_since_last_report / (i - self.last_iteration_num)
                time_estimate = datetime.timedelta(seconds = (total - i) * sec_per_iter)
                time_estimate = str(time_estimate).split('.')[0]
                message = f'Completed {i}/{total}. Estimated Time Remaining: {time_estimate}.'
                if self.report_level == logging.DEBUG:
                    self.debug(message)
                elif self.report_level == logging.INFO:
                    self.info(message)
                elif self.report_level == logging.WARNING:
                    self.warning(message)
                elif self.report_level == logging.ERROR:
                    self.error(message)
                elif self.report_level == logging.CRITICAL:
                    self.critical(message)
                self.last_iteration_num = i
            self.last_report_time = time.time()
        return

    def format_log(self, message, levelname):
        return f'{self.name} - {levelname} - {time.strftime("%H:%M:%S")}: {message}'

    def info(self, message):
        if logging.INFO >= self.log_level:
            print(self.format_log(message, 'INFO'))

    def debug(self, message):
        if logging.DEBUG >= self.log_level:
            print(self.format_log(message, 'DEBUG'))

    def warning(self, message):
        if logging.WARNING >= self.log_level:
            print(self.format_log(message, 'WARNING'))

    def error(self, message):
        if logging.ERROR >= self.log_level:
            print(self.format_log(message, 'ERROR'))

    def critical(self, message):
        if logging.CRITICAL >= self.log_level:
            print(self.format_log(message, 'CRITICAL'))

class BaseRemoteClass:
    def __init__(
            self,
            name,
            log_level,
            name_idx=None,
            ):
        if name_idx is None:
            self.name = name
        else:
            self.name = f'{name}_{name_idx}'
        self.name_idx = name_idx
        self.log_level = log_level
    
    def get_attr(self, attr_name):
        return getattr(self, attr_name)

    def format_log(self, message, levelname):
        return f'{self.name} - {levelname} - {time.strftime("%H:%M:%S")}: {message}'

    def info(self, message):
        if logging.INFO >= self.log_level:
            print(self.format_log(message, 'INFO'))

    def debug(self, message):
        if logging.DEBUG >= self.log_level:
            print(self.format_log(message, 'DEBUG'))

    def warning(self, message):
        if logging.WARNING >= self.log_level:
            print(self.format_log(message, 'WARNING'))

    def error(self, message):
        if logging.ERROR >= self.log_level:
            print(self.format_log(message, 'ERROR'))

    def critical(self, message):
        if logging.CRITICAL >= self.log_level:
            print(self.format_log(message, 'CRITICAL'))
