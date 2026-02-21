package com.uav.eval.service.impl;

import com.uav.eval.entity.Dataset;
import com.uav.eval.dao.DatasetDao;
import com.uav.eval.service.IDatasetService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

/**
 * <p>
 * 无人机影像数据集表 服务实现类
 * </p>
 *
 * @author wenzhu
 * @since 2026-02-21
 */
@Service
public class DatasetServiceImpl extends ServiceImpl<DatasetDao, Dataset> implements IDatasetService {

}
