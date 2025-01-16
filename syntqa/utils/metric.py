import math
import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import ast


def text_to_list(text): # for wikiTQ
    """
    텍스트 형태의 리스트를 실제 리스트로 변환.
    
    :param text: 문자열 (예: "['Siim Ennemuist', 'Andri Aganits']")
    :return: 리스트 (예: ['Siim Ennemuist', 'Andri Aganits'])
    """
    return ast.literal_eval(text)


def normalize_fraction(text):
    """
    텍스트를 분수 형태로 표준화 (텍스트 기반 처리).
    :param text: str, 입력 텍스트
    :return: str, 표준화된 분수 텍스트
    """
    # LaTeX 분수 표현을 정규식으로 변환
    latex_fraction_pattern = r"\\frac{(\d+)}{(\d+)}"
    match = re.search(latex_fraction_pattern, text)
    if match:
        numerator, denominator = match.groups()
        return f"{numerator}/{denominator}"
    return text  # LaTeX 형식이 아니면 원본 반환

def normalize_number(text):
    """
    쉼표 제거 및 숫자를 표준화.
    :param text: str, 입력 텍스트
    :return: str, 변환된 텍스트
    """
    return text.replace(",", "").strip()

def normalize_time_format(text):
    """
    시간 표현을 정규화하여 포함 비교 가능하게 처리.
    P.M. 및 A.M.과 같은 특수한 시간 표현을 처리.
    """
    return text.replace("P.M.", "P.M").replace("A.M.", "A.M").strip()


def parse_numeric_value(text):
    """
    텍스트를 숫자 값으로 파싱하려고 시도합니다.
    분수, 소수, 퍼센트, LaTeX 분수 등을 처리합니다.
    :param text: str, 입력 텍스트
    :return: float 또는 None, 파싱된 숫자 값 또는 실패 시 None
    """
    text = text.strip()
    text = text.replace(",", "")  # 쉼표 제거

    # 퍼센트 처리
    if text.endswith('%'):
        try:
            value = float(text[:-1]) / 100.0
            return value
        except ValueError:
            pass

    # LaTeX 분수 처리
    match = re.match(r"\\frac{(-?\d+)}{(-?\d+)}", text)
    if match:
        numerator, denominator = match.groups()
        try:
            value = float(numerator) / float(denominator)
            return value
        except (ValueError, ZeroDivisionError):
            pass

    # 일반 분수 처리
    if '/' in text:
        try:
            numerator, denominator = text.split('/')
            value = float(numerator) / float(denominator)
            return value
        except (ValueError, ZeroDivisionError):
            pass

    # 소수 또는 정수 처리
    try:
        value = float(text)
        return value
    except ValueError:
        pass

    # 파싱 실패 시 None 반환
    return None


def check_match(pred, ans):
    """
    :param pred, ans : 예측값과 정답갑. 이 둘이 일치하는지 여부를 판단할 것이고 판단을 위해 둘의 format을 맞춰주고 체크함
    """

    pred = pred.lower()
    ans = ans.lower()

    # 포함 비율 계산
    if pred in ans:
        return True
    elif ans in pred:
        return True
    else:
        # 따옴표 제거 후 한 번 더 비교
        ans_cleaned = ans.replace("'", "")
        pred_cleaned = pred.replace("'", "")
        if ans_cleaned in pred_cleaned:
            return True
        elif pred_cleaned in ans_cleaned:
            return True
        else:
            # 분수 비교 추가
            ans_fraction = normalize_fraction(ans)
            pred_fraction = normalize_fraction(pred)
            if ans_fraction in pred_fraction:
                return True
            elif pred_fraction in ans_fraction:
                return True
            else:
                # 콤마 제거하는 숫자 비교 추가 (ex. 12,345 -> 12345)
                ans_number = normalize_number(ans)
                pred_number = normalize_number(pred)
                if ans_number in pred_number:
                    return True
                elif pred_number in ans_number:
                    return True
                else:
                    # 시간 표현 비교 추가
                    ans_time = normalize_time_format(ans)
                    pred_time = normalize_time_format(pred)
                    if ans_time in pred_time:
                        return True
                    else:
                        # 쌍따옴표 제거 후 한 번 더 비교
                        ans_cleaned_dubble = ans.replace('''"''', "").replace(".", "").strip()
                        pred_cleaned_dubble = pred.replace('''"''', "").replace(".", "").strip()
                        if ans_cleaned_dubble in pred_cleaned_dubble:
                            return True
                        elif pred_cleaned_dubble in ans_cleaned_dubble:
                            return True
                        else:
                            # the 제거 후 한 번 더 비교
                            ans_cleaned_the = ans.replace('''"''', "").replace(".", "").replace("the", "").strip()
                            pred_cleaned_the = pred.replace('''"''', "").replace(".", "").replace("the", "").strip()
                            if ans_cleaned_the in pred_cleaned_the:
                                return True
                            elif pred_cleaned_the in ans_cleaned_the:
                                return True    
                            else:
                                # 숫자 값 비교 추가
                                ans_value = parse_numeric_value(ans)
                                pred_value = parse_numeric_value(pred)
                                if ans_value is not None and pred_value is not None:
                                    if math.isclose(ans_value, pred_value, rel_tol=1e-6, abs_tol=1e-12):
                                        return True
                                    else:
                                        return False
                                else:
                                    return False


