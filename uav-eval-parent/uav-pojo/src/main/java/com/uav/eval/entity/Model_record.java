package com.uav.eval.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.Version;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import java.io.Serializable;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 * 模型测评记录表
 * </p>
 *
 * @author wenzhu
 * @since 2026-02-21
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("tbl_model_record")
public class Model_record implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 主键ID
     */
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    /**
     * 关联的数据集ID
     */
    private Long dataset_id;

    /**
     * 模型架构(如yolov8-drone-mamba)
     */
    private String model_name;

    /**
     * mAP精度得分(如0.852)
     */
    private Double map_val;

    /**
     * 推理速度FPS
     */
    private Integer fps;

    /**
     * 乐观锁版本号
     */
    @Version
    private Integer version;

    /**
     * 逻辑删除(0正常, 1删除)
     */
    @TableLogic
    private Integer deleted;


}
