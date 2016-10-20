package main;

import me.xiaosheng.lda.HanLDA;

public class HanLDATest {

	public static void test1() throws Exception {
		//训练语料，产生 LDA 模型
		HanLDA.train("data/mini", 10, "model/lda1.model", true);
		//预测指定文档的主题分布
		System.out.println("---军事_510.txt 的主题分布预测---");
		HanLDA.inference("model/lda1.model", "data/mini/军事_510.txt", true);
	}
	
	public static void test2() throws Exception {
		//训练语料，产生 LDA 模型
		HanLDA.train("data/mini", 10, "model/lda2.model", true);
		HanLDA hanLDA = new HanLDA("model/lda2.model");
		//预测指定文档的主题分布
		System.out.println("---体育_1200.txt 的主题分布预测---");
		hanLDA.inference("data/mini/体育_1200.txt", true);
		System.out.println("\n---旅游_1620.txt 的主题分布预测---");
		hanLDA.inference("data/mini/旅游_1620.txt", true);
	}
	
	public static void main(String[] args) {
		try {
			//test1();
			test2();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
