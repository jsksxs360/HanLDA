package me.xiaosheng.lda;

import java.io.IOException;
import java.util.Map;

import com.hankcs.lda.Corpus;
import com.hankcs.lda.LdaGibbsSampler;
import com.hankcs.lda.LdaUtil;

public class HanLDA {
	LDAModel ldaModel;
	
	public HanLDA(String modelFilePath) {
		try {
			ldaModel = new LDAModel(modelFilePath);
		} catch (IOException e) {
			e.printStackTrace();
			ldaModel = null;
		}
	}
	
	public double[] inference(String documentFilePath, boolean printResult) throws IOException {
		if (ldaModel == null)
			return null;
		int[] document = Corpus.loadDocument(documentFilePath, ldaModel.getVocabulary());
		if (document.length == 0)
			return null;
		double[] tp = LdaGibbsSampler.inference(ldaModel.getPhiMatrix(), document);
        if (printResult) {
        	for(int i = 0; i < tp.length; i++) {
        		System.out.println("topic" + i + "-----" + tp[i]);
        	}
        }
        return tp;
	}
	
	/**
	 * 训练模型
	 * @param trainFolderPath 训练语料文件所在目录
	 * @param trainTopicNum 训练的主题个数
	 * @param saveModelFilePath 模型文件保存路径
	 * @param printLDAModel 是否打印产生的模型
	 * @throws Exception
	 */
	public static void train(String trainFolderPath, int trainTopicNum, String saveModelFilePath, boolean printLDAModel) throws Exception {
		//载入语料，预处理
		Corpus corpus = Corpus.load(trainFolderPath);
        if (corpus == null)
        	throw new Exception("训练语料载入失败");
        //创建 LDA 采样器
        LdaGibbsSampler ldaGibbsSampler = new LdaGibbsSampler(corpus.getDocument(), corpus.getVocabularySize());
        //训练指定个数的主题
        if (trainTopicNum < 1)
			throw new Exception("训练的主题个数至少为1");
        ldaGibbsSampler.gibbs(trainTopicNum);
        //phi 矩阵是唯一有用的东西，可以用 LdaUtil 来展示最终的结果
        double[][] phi = ldaGibbsSampler.getPhi();
        //存储产生的 LDA 模型
        LDAModel.saveModelFile(saveModelFilePath, phi, corpus.getVocabulary());
        if (printLDAModel) { //展示产生的模型，每个主题展示 10 个词汇(按出现概率降序)
        	Map<String, Double>[] topicMap = LdaUtil.translate(phi, corpus.getVocabulary(), 10);
        	LdaUtil.explain(topicMap);
        }
        //存储LDA展示文件，文件名为 modelFileName.txt，每个主题展示 20 个词汇(按出现概率降序)
        LDAModel.saveLDAShowFile(saveModelFilePath + ".txt", phi, corpus.getVocabulary(), 20);
	}
	
	/**
	 * 预测文档的主题分布
	 * @param modelFilePath 模型文件路径
	 * @param documentFilePath 要预测的文档路径
	 * @param printResult 是否打印结果
	 * @return 主题分布数组
	 * @throws IOException
	 */
	public static double[] inference(String modelFilePath, String documentFilePath, boolean printResult) throws IOException {
		//读取训练好的 LDA 模型
		LDAModel ldaModel = new LDAModel(modelFilePath);
		//进行文档的主题分布预测
		int[] document = Corpus.loadDocument(documentFilePath, ldaModel.getVocabulary());
		if (document.length == 0)
			return null;
        double[] tp = LdaGibbsSampler.inference(ldaModel.getPhiMatrix(), document);
        if (printResult) {
        	for(int i = 0; i < tp.length; i++) {
        		System.out.println("topic" + i + "-----" + tp[i]);
        	}
        }
        return tp;
	}
	
	public void changeModel(String newModelFilePath) throws IOException {
		ldaModel = new LDAModel(newModelFilePath);
	}
}
