package com.uav.eval.controller;


import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.uav.eval.Result;
import com.uav.eval.entity.Dataset;
import com.uav.eval.service.IDatasetService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * <p>
 * 无人机影像数据集表 前端控制器
 * </p>
 *
 * @author wenzhu
 * @since 2026-02-21
 */
@RestController // 代表这是一个提供 REST API 的控制器，所有方法返回的数据都会转成 JSON
@RequestMapping("/datasets") // 统一的路由前缀
public class DatasetController {

    @Autowired
    private IDatasetService datasetService; // 直接注入 MP 为我们准备好的大血包

    /**
     * 高级实战：带模糊搜索的分页查询
     * 场景：你在前端网页想要查包含 "VisDrone" 或 "RSOD" 的数据集，并且每页显示 10 条。
     */
    @GetMapping("/page")
    public Result getPage(@RequestParam(defaultValue = "1") Integer current,
                          @RequestParam(defaultValue = "10") Integer size,
                          String name) { // name 是前端传过来的搜索关键字，可能为 null

        // 1. 构造 MyBatisPlus 的分页对象 IPage
        IPage<Dataset> page = new Page<>(current, size);

        // 2. 构造查询条件 LambdaQueryWrapper
        LambdaQueryWrapper<Dataset> lqw = new LambdaQueryWrapper<>();

        // 核心知识点：null 判定！如果前端没传 name，这个条件就不拼接；如果传了，就执行 LIKE 模糊查询
        lqw.like(name != null && !name.isEmpty(), Dataset::getName, name);

        // 我们还可以加一个排序条件，比如按图片数量降序排列，把数据量大的排在前面
        lqw.orderByDesc(Dataset::getImage_count);

        // 3. 执行分页查询，结果会自动封装进 page 对象中
        datasetService.page(page, lqw);

        // 4. 返回统一格式给前端
        return new Result(200, "查询成功", page);
    }

    /**
     * 录入新的数据集信息 (新增)
     */
    @PostMapping
    public Result save(@RequestBody Dataset dataset) {
        // 调用 IService 的 save 方法直接保存
        boolean flag = datasetService.save(dataset);
        return new Result(flag ? 200 : 500, flag ? "保存成功" : "保存失败", null);
    }

    /**
     * 安全删除数据集 (逻辑删除)
     */
    @DeleteMapping("/{id}")
    public Result delete(@PathVariable Long id) {
        // 调用 removeById，因为配了 @TableLogic，底层会自动执行 UPDATE 语句变更为已删除状态
        boolean flag = datasetService.removeById(id);
        return new Result(flag ? 200 : 500, flag ? "删除成功" : "删除失败", null);
    }

    /**
     * 更新数据集信息 (触发乐观锁)
     */
    @PutMapping
    public Result update(@RequestBody Dataset dataset) {
        // 注意：前端传过来的 dataset 必须包含 id 和 version 字段，这样 updateById 才能触发乐观锁机制
        boolean flag = datasetService.updateById(dataset);
        return new Result(flag ? 200 : 500, flag ? "更新成功" : "更新失败，数据已被其他人修改", null);
    }
}

