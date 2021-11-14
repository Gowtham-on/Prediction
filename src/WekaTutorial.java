import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;

public class WekaTutorial
{

  public static void main(String[] args) throws Exception
  {
    DataSource trainSource = new DataSource("F:/Gowtham/Weka/iris.arff"); // training
    Instances trainData = trainSource.getDataSet();

    DataSource testSource = new DataSource("F:/Gowtham/Weka/iris-copy.arff");
    Instances testData = testSource.getDataSet();

    if (trainData.classIndex() == -1)
    {
      trainData.setClassIndex(trainData.numAttributes() - 1);
    }

    if (testData.classIndex() == -1)
    {
      testData.setClassIndex(testData.numAttributes() - 1);
    }    

    String[] options = weka.core.Utils.splitOptions("weka.filters.unsupervised.attribute.StringToWordVector -R first-last -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -M 1 "
            + "-tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \" \\r\\n\\t.,;:\\\'\\\"()?!\"");

    Remove remove = new Remove();
//    remove.setOptions(options);
    remove.setInputFormat(trainData);

    NominalToBinary filter = new NominalToBinary(); 

    NaiveBayes nb = new NaiveBayes();

    FilteredClassifier fc = new FilteredClassifier();
    fc.setFilter(filter);
    fc.setClassifier(nb);
    // train and make predictions
    fc.buildClassifier(trainData);

    for (int i = 0; i < testData.numInstances(); i++)
    {
      double pred = fc.classifyInstance(testData.instance(i));
      System.out.print("ID: " + testData.instance(i).value(0));
      System.out.print(", actual: " + testData.classAttribute().value((int) testData.instance(i).classValue()));
      System.out.println(", predicted: " + testData.classAttribute().value((int) pred));
    }

  }

}