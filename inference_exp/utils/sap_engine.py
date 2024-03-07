import os
from colorama import Fore, Style

class sAP_Engine:
    current_index = 0

    def __init__(self):
        None


    def get_index_jump(self, inf_time_ms, data_interval_ms):
        jump = int(inf_time_ms / data_interval_ms)
        if inf_time_ms%data_interval_ms != 0:
            jump += 1
        return jump


    def get_index_and_jump(self, inf_time_ms, data_interval_ms):
        jump = self.get_index_jump(inf_time_ms, data_interval_ms)
        return self.current_index, jump


    def get_index_and_jump_with_skip(self, inf_time_ms, data_interval_ms):
        cur = self.current_index
        jump = self.get_index_jump(inf_time_ms, data_interval_ms)
        self.current_index += jump
        return cur, jump


if __name__ == "__main__":
    sap_engine = sAP_Engine()
    sap_engine.current_index = 10
    cur, jump = sap_engine.get_index_and_jump(100, 10)
    print(f"cur: {cur}, jump: {jump}")
    cur, jump = sap_engine.get_index_and_jump_with_skip(100, 10)
    print(f"cur before skip: {cur}, jump: {jump}")
    print(f"cur after skip by jump: {sap_engine.current_index}")

