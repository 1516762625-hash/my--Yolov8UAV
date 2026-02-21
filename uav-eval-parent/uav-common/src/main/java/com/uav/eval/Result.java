package com.uav.eval;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Result {
    private Integer code; // 状态码，比如 200 代表成功，500 代表失败
    private String msg;   // 提示信息，比如 "查询成功"
    private Object data;  // 真正要返回给前端的数据（比如分页对象、数据集列表等）
}
