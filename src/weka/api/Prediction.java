package weka.api;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Prediction {

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("F:/Gowtham/Weka/iris.arff");
		Instances trainDataset = source.getDataSet();
		
		trainDataset.setClassIndex(trainDataset.numAttributes()-1);
		
		int numClasses = trainDataset.numClasses();
		
		for (int i = 0; i < numClasses; i++) {
			String classValue = trainDataset.classAttribute().value(i);
			System.out.println("Class value " + i + "is " + classValue);
		}
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(trainDataset);
		
		//load new dataset
		DataSource source1 = new DataSource("F:/Gowtham/Weka/iris-copy.arff");
		Instances testDataset = source.getDataSet();
		
	}

}
