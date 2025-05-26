import torch
from time import time
from datetime import timedelta

LANG_REGEX = {
    "ko": r"[ㄱ-ㅎㅏ-ㅣ가-힣]+",
    "ja": r"[ぁ-ゔァ-ヴー々〆〤ｧ-ﾝﾞﾟ]+",
    "zh": r"[\u4e00-\u9fff]+",
}
REGEX = r"[ㄱ-ㅎㅏ-ㅣ가-힣ぁ-ゔァ-ヴー々〆〤ｧ-ﾝﾞﾟ\u4e00-\u9fff]+"


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))

def print_number_of_parameters(model):
    print(f"""{sum([p.numel() for p in model.parameters()]):,}""")
