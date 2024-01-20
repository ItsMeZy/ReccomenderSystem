using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;

class CollaborativeFilteringSimple
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }

    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }

    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

        ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

        Console.WriteLine("=============== Evaluating the model ===============");
        var prediction = model.Transform(testDataView);
        var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
        Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
        Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
    }

    static (IDataView training, IDataView test) LoadData(MLContext mlContext)
    {
        var trainingDataPath = Path.Combine(Environment.CurrentDirectory+"\\..\\..\\..\\train.csv");
        var testDataPath = Path.Combine(Environment.CurrentDirectory+"\\..\\..\\..\\test.csv");

        IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
        IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');

        return (trainingDataView, testDataView);
    }

    static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
    {
        IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

        var options = new MatrixFactorizationTrainer.Options
        {
            MatrixColumnIndexColumnName = "userIdEncoded",
            MatrixRowIndexColumnName = "movieIdEncoded",
            LabelColumnName = "Label",
            NumberOfIterations = 100,
            ApproximationRank = 100
        };

        var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

        Console.WriteLine("=============== Training the model ===============");
        ITransformer model = trainerEstimator.Fit(trainingDataView);

        return model;
    }
}