package com.uav.eval;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.generator.AutoGenerator;
import com.baomidou.mybatisplus.generator.config.DataSourceConfig;
import com.baomidou.mybatisplus.generator.config.GlobalConfig;
import com.baomidou.mybatisplus.generator.config.PackageConfig;
import com.baomidou.mybatisplus.generator.config.StrategyConfig;

public class CodeGenerator {
    public static void main(String[] args) {
        // 1. 获取代码生成器的对象
        AutoGenerator autoGenerator = new AutoGenerator();

        // 2. 设置数据库相关配置
        DataSourceConfig dataSource = new DataSourceConfig();
        dataSource.setDriverName("com.mysql.cj.jdbc.Driver");
        dataSource.setUrl("jdbc:mysql://localhost:3306/uav_eval_db?serverTimezone=Asia/Shanghai&useUnicode=true&characterEncoding=utf-8");
        dataSource.setUsername("root");
        dataSource.setPassword("123456"); // 改成你自己的 MySQL 密码
        autoGenerator.setDataSource(dataSource);

        // 3. 设置全局配置
        GlobalConfig globalConfig = new GlobalConfig();
        // 设置代码生成位置：直接生成到当前 uav-web 模块下
        globalConfig.setOutputDir(System.getProperty("user.dir") + "/uav-web/src/main/java");
        globalConfig.setOpen(false); // 生成完毕后是否自动打开目录
        globalConfig.setAuthor("wenzhu"); // 设置作者
        globalConfig.setFileOverride(true); // 是否覆盖已有文件
        globalConfig.setMapperName("%sDao"); // 我们习惯叫 Dao，把默认的 Mapper 改名
        globalConfig.setIdType(IdType.ASSIGN_ID); // 全局配置雪花算法 ID
        autoGenerator.setGlobalConfig(globalConfig);

        // 4. 设置包名相关配置
        PackageConfig packageInfo = new PackageConfig();
        packageInfo.setParent("com.uav.eval"); // 设置生成的父包名
        packageInfo.setEntity("entity"); // 实体类包名
        packageInfo.setMapper("dao"); // 数据层包名
        autoGenerator.setPackageInfo(packageInfo);

        // 5. 策略设置 (核心：把前面学的知识点全部映射上去)
        StrategyConfig strategyConfig = new StrategyConfig();
        strategyConfig.setInclude("tbl_dataset", "tbl_model_record"); // 指定要生成的表名
        strategyConfig.setTablePrefix("tbl_"); // 去掉表前缀，让生成的类叫 Dataset 而不是 TblDataset
        strategyConfig.setRestControllerStyle(true); // 启用 Rest 风格，生成的 Controller 会带 @RestController 注解
        strategyConfig.setVersionFieldName("version"); // 映射乐观锁字段
        strategyConfig.setLogicDeleteFieldName("deleted"); // 映射逻辑删除字段
        strategyConfig.setEntityLombokModel(true); // 自动给实体类加上 @Data 等 Lombok 注解
        autoGenerator.setStrategy(strategyConfig);

        // 6. 执行生成操作
        autoGenerator.execute();
        System.out.println("====== 恭喜！UAV 测评系统底层代码生成完毕！======");
    }
}
