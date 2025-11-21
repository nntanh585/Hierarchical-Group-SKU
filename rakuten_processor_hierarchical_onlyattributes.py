import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# --- HELPER FUNCTIONS ---

def _read_json(json_file: str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def _clean_text(text: str) -> str:
    """Làm sạch văn bản cơ bản."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text) # Remove HTML tags
    text = re.sub(r"\[.*?\]", " ", text) # Remove text in square brackets
    text = re.sub(r"[^\w\s]", " ", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

def _extract_adaptive_attributes(attributes_list: list) -> dict:
    """
    Trích xuất thích ứng (adaptive) TẤT CẢ các cặp key-value.
    """
    result_dict = {}
    if not attributes_list:
        return result_dict

    for attr in attributes_list:
        key = None
        value = None

        # Case 1: Cấu trúc 'required_attributes'
        if 'name' in attr and 'value' in attr:
            key_raw = attr.get('name')
            value_list = attr.get('value', [])
            if key_raw and value_list and value_list[0]:
                key = _clean_text(key_raw)
                value = _clean_text(value_list[0])
                unit = attr.get('unit')
                if unit:
                    value = f"{value} {unit}"

        # Case 2: Cấu trúc 'attribute_simples'
        elif 'attributes' in attr and 'attribute_values' in attr:
            key_raw = attr.get('attributes', {}).get('name')
            value_raw = attr.get('attribute_values', {}).get('name')
            if key_raw and value_raw:
                key = _clean_text(key_raw)
                value = _clean_text(value_raw)

        if key and value:
            result_dict[key] = value

    return result_dict

def _transform_data_from_json(json_file: str) -> pd.DataFrame:
    """
    Đọc file JSON và làm phẳng dữ liệu.
    """
    full_json_data = _read_json(json_file)
    products_list = full_json_data.get('data', []) 

    result_list = []

    for product_data in products_list:
        # 1. Thông tin Cha
        parent_id = product_data.get('id')
        parent_name = _clean_text(product_data.get('name'))
        parent_category = _clean_text(product_data.get('category', {}).get('name_en'))
        parent_short_desc = _clean_text(product_data.get('short_description'))
        
        # Xử lý an toàn cho jan_info
        jan_info = product_data.get('jan_info', {})
        jan_code = ""
        if isinstance(jan_info, dict):
            jan_code = str(jan_info.get('reason_no_code') or jan_info.get('code') or "")

        # 2. Lặp qua các Con (SKUs)
        skus_data = product_data.get('product_skus', {}).get('data', [])
        if skus_data:
            for sku in skus_data:
                req_attrs_list = sku.get('required_attributes', [])
                simple_attrs_list = sku.get('attribute_simples', {}).get('data', [])

                attributes_dict = _extract_adaptive_attributes(req_attrs_list)
                attributes_dict.update(_extract_adaptive_attributes(simple_attrs_list))

                product_info = {
                    "product_id": parent_id,
                    "product_name": parent_name,
                    "category_name": parent_category,
                    "short_description": parent_short_desc,
                    "seller_sku": _clean_text(sku.get("seller_sku")),
                    "attributes": attributes_dict,
                    "jan_infor": jan_code
                }
                result_list.append(product_info)

    return pd.DataFrame(result_list) if result_list else pd.DataFrame()

# --- 2 HÀM GET TEXT RIÊNG BIỆT (CUSTOM THEO YÊU CẦU) ---

def _get_master_text(product_row: pd.Series) -> str:
    """
    Tạo chuỗi text cho MASTER clustering.
    Mục tiêu: Gom các biến thể (Màu/Size) về chung 1 nhóm.
    Chiến lược: Chỉ lấy thông tin chung (Tên, Hãng, Model, Danh mục, JAN).
                LOẠI BỎ SKU và các thuộc tính biến đổi.
    """
    name = product_row.get('product_name', '')
    category = product_row.get('category_name', '')
    short_description = product_row.get('short_description', '')
    jan_infor = product_row.get('jan_infor', '')
    sku = product_row.get('seller_sku', '')

    # Lấy attributes để tìm Brand/Model, nhưng không lấy hết
    attributes_dict = product_row.get('attributes', {})
    
    # Cố gắng tìm Brand và Model trong đống attributes hỗn độn
    brand = ""
    model = ""
    
    # Các từ khóa phổ biến để nhận diện Brand/Model (đã clean text)
    for k, v in attributes_dict.items():
        if 'brand' in k or 'ブランド' in k: # Brand name
            brand = v
        if 'model' in k or '型番' in k: # Model number
            model = v

    # Cấu trúc prompt cho Master (Không có SKU, không có Color/Size cụ thể)
    text = f"Product: {name}\nSKU: {sku}\nBrand: {brand}\nModel: {model}\nCategory: {category}\nJAN: {jan_infor}\nDescription: {short_description}"
    return text

def _get_variant_text(product_row: pd.Series) -> str:
    """
    Tạo chuỗi text cho VARIANT clustering.
    Mục tiêu: Phân biệt sự khác nhau giữa các SKU trong cùng 1 nhóm Master.
    Chiến lược: Tập trung vào SKU và TOÀN BỘ thuộc tính chi tiết.
    """
    # Vẫn cần tên để giữ ngữ cảnh (ví dụ phân biệt Áo vs Quần nếu lỡ Master gom sai)
    # Nhưng trọng số chính sẽ nằm ở phần Details và SKU
    name = product_row.get('product_name', '')
    attributes_dict = product_row.get('attributes', {})

    # Loại bỏ các key không mong muốn (như user yêu cầu)
    remove_keys = {'総個数', '総重量', '総容量'}
    filtered_attrs = {k: v for k, v in attributes_dict.items() if k not in remove_keys}
    
    # Tạo chuỗi chi tiết thuộc tính
    details = " | ".join([f"{k}: {v}" for k, v in filtered_attrs.items()])

    # Cấu trúc prompt cho Variant (SKU và Details là quan trọng nhất)
    text = f"Details: {details}\nContext: {name}"
    return text