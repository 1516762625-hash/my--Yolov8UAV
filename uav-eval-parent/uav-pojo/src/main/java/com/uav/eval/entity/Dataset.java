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
 * 无人机影像数据集表
 * </p>
 *
 * @author wenzhu
 * @since 2026-02-21
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("tbl_dataset")
public class Dataset implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 主键ID(雪花算法)
     */
    @TableId(value = "id", type = IdType.ASSIGN_ID)
    private Long id;

    /**
     * 数据集名称(如VisDrone, RSOD, SSDD)
     */
    private String name;

    /**
     * 标注格式(如YOLO, VOC)
     */
    private String format;

    /**
     * 图片数量
     */
    private Integer image_count;

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
