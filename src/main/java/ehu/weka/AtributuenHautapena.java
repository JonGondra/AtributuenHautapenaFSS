package ehu.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.ArrayList;

public class AtributuenHautapena {
    private static final AtributuenHautapena instance = new AtributuenHautapena();

    public static AtributuenHautapena getInstance(){
        return instance;
    }

    private AtributuenHautapena(){

    }

    public Instances datuakKargatu(String path) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        return data;
    }

    private Instances randomize(Instances data) throws Exception {
        Randomize filter = new Randomize();
        filter.setInputFormat(data);
        Instances randomData = Filter.useFilter(data,filter);
        randomData.setClassIndex(randomData.numAttributes()-1);
        return randomData;
    }

    private Instances splitData(Instances data, double percent, boolean invert) throws Exception {
        RemovePercentage filterRemove = new RemovePercentage();
        filterRemove.setInputFormat(data);
        filterRemove.setPercentage(percent);
        filterRemove.setInvertSelection(invert);
        Instances split = Filter.useFilter(data,filterRemove);
        split.setClassIndex(split.numAttributes()-1);
        return split;
    }

    public Classifier sailkatzailea(){
        return new NaiveBayes();
    }

    public Evaluation holdOut(Instances data, double percent) throws Exception {
        //Randomize
        Instances randomData = randomize(data);

        //Split Data
        Instances test = splitData(randomData,percent,false);
        Instances train = splitData(randomData,percent,true);

        //Sailkatzailea entrenatu
        Classifier model = sailkatzailea();
        model.buildClassifier(train);

        //Ebaluatu
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model,test);
        return eval;
    }

    public void serialization(String path, Classifier model) throws Exception {
        // serialize model
        weka.core.SerializationHelper.write(path,model);
    }

    public Classifier deserialize(String path) throws Exception {
        // deserialize model
        return (Classifier) weka.core.SerializationHelper.read(path);
    }

    public Instances selection(Instances data) throws Exception {
        AttributeSelection filterSelection = new AttributeSelection();
        filterSelection.setInputFormat(data);
        Instances selection = Filter.useFilter(data,filterSelection);
        selection.setClassIndex(selection.numAttributes()-1);
        return selection;
    }

    public Instances replaceMissingValues(Instances data) throws Exception {
        ReplaceMissingValues filterReplace = new ReplaceMissingValues();
        filterReplace.setInputFormat(data);
        Instances replaced = Filter.useFilter(data,filterReplace);
        replaced.setClassIndex(replaced.numAttributes()-1);
        return replaced;
    }

    public String[] atributuLista(Instances data) throws Exception {
        String[] emaitza = new String[data.numAttributes()];
        for(int i=0; i<emaitza.length; i++){ emaitza[i] = data.attribute(i).name(); }
        return emaitza;
    }

    public int[] arrayConverter(ArrayList<Integer> integers){
        int[] emaitza = new int[integers.size()];
        for (int i=0; i < emaitza.length; i++)
        {
            emaitza[i] = integers.get(i).intValue();
        }
        return emaitza;
    }
}
