package ehu.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;

public class MainAtributuenHautapena {
    public static void main(String[] args) throws Exception {
        if (args.length != 6) {
            System.out.println("Atazak:");
            System.out.println("\t 1. Gainbegiratutako datuak emanda, eredu optimoa lortu. ");
            System.out.println("\t 2. Eredua aplikatu etiketatuta ez dagoen test multzoari (test blind) eta iragarpenak lortu. ");
            System.out.println("\t 3. Aurre-prozesua: Train/test multzoak missing values daukate? Egokia litzateke galdutako balioak inputatzea? " +
                    "\n\tErrepikatu prozesua inputatutako datuekin eta emaitzak aztertu. ");

            System.out.println("\nArgumentuak:");
            System.out.println("\t 1. Datu sortaren kokapena (path) .arff  formatuan (input). Aurre-baldintza: klasea azken atributuan egongo da.");
            System.out.println("\t 2. NB.model: eredua gordetzeko path");
            System.out.println("\t 3. Blind-test sortaren kokapena (path) .arff  formatuan (input).");
            System.out.println("\t 4. predictions.txt: iragarpenak gordetzeko path");
            System.out.println("\t 5. aurre.model: aurre-prozesua aplikatu ostean eredua gordetzeko path");
            System.out.println("\t 6. predictionsAurre.txt: aurre-prozesua aplikatu ostean iragarpenak gordetzeko path");
            System.exit(0);
        }

        AtributuenHautapena fss = AtributuenHautapena.getInstance();
        Instances data = fss.datuakKargatu(args[0]);


        //1. Eredu iragarle optimoa sortu eta gorde
        Instances selection = iragarleOptimoa(data,args[1]);


        //2. Test multzoaren iragarpenak egin
        Instances test = fss.datuakKargatu(args[2]);
        Classifier model = fss.deserialize(args[1]);
        iragarpenak(selection,test,model,args[3]);


        //3. Aurre-prozesua:
        Instances dataR = fss.replaceMissingValues(data);
        Instances testR = fss.replaceMissingValues(test);

        //3.1.Iragarle optimoa
        Instances selectionR =iragarleOptimoa(dataR,args[4]);

        //3.2.Iragarpenak
        Classifier modelR = fss.deserialize(args[4]);
        iragarpenak(selectionR,testR,modelR,args[5]);

    }


    public static Instances iragarleOptimoa(Instances data, String pathModel) throws Exception {
        //1. Eredu iragarle optimoa sortu eta gorde
        AtributuenHautapena fss = AtributuenHautapena.getInstance();
        Instances selection = fss.selection(data);
        Evaluation holdOut = fss.holdOut(selection,70);

        System.out.println("Atributu kopurua: "+selection.numAttributes());
        System.out.println("\n"+holdOut.toMatrixString()+"\n");
        System.out.println("F-score: "+holdOut.weightedFMeasure()+"\n");

        Classifier cls = fss.sailkatzailea();
        cls.buildClassifier(selection);
        fss.serialization(pathModel,cls);

        return selection;
    }


    public static void iragarpenak(Instances data, Instances test, Classifier model, String path) throws Exception {
        //2. Test multzoaren iragarpenak egin
        AtributuenHautapena fss = AtributuenHautapena.getInstance();
        if(!data.equalHeaders(test)){
            System.out.println("test1 : "+test.numAttributes());
            System.out.println("data : "+test.numAttributes());
            int i = 0;
            for(Attribute attribute : Collections.list(test.enumerateAttributes())){
                if(!Collections.list(data.enumerateAttributes()).contains(attribute)){
                    test.deleteAttributeAt(attribute.index()-i);
                    i++;
                }
            }
            /*
            String[] atribLista = fss.atributuLista(data);
            ArrayList<Integer> indizeak = new ArrayList<Integer>();
            int j=0;
            for(int i=0; i<atribLista.length;){
                if(atribLista[i].equals(test.attribute(j).name())){
                    j++;
                    i++;
                }
                else{
                    indizeak.add(j);
                    j++;
                }
            }
            int[] array = fss.arrayConverter(indizeak);
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(array);
            remove.setInvertSelection(true);
            remove.setInputFormat(test);
            test = Filter.useFilter(test,remove);
            test.setClassIndex(test.numAttributes()-1);
            System.out.println("test2 : "+test.numAttributes());

             */
        }

        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(model,test);

        FileWriter fw = new FileWriter(path);
        fw.write("Exekuzio data: "+java.time.LocalDateTime.now().toString()+"\n");
        fw.write("\n-- Test Set -- \n");
        fw.write("Instantzia\tActual\tPredicted Errorea\n");
        for(int i=0; i< eval.predictions().size();i++){
            double actual = eval.predictions().get(i).actual();
            double predicted = eval.predictions().get(i).predicted();

            fw.write("\t"+(i+1)+"\t"+test.instance(i).stringValue(test.classIndex())+ "\t"+test.attribute(test.classIndex()).value((int) predicted));

            if(actual!=predicted && !Double.isNaN(actual)){
                fw.write("\t     +");
            }
            fw.write("\n");
        }
        fw.close();
    }
}
