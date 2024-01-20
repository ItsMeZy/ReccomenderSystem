
using Accord.MachineLearning.Rules;
using System.Collections.Generic;

class Program
{
    static void Main(string[] args)
    {
        Random r = new Random(100);
        List<SortedSet<int>> dataset = new List<SortedSet<int>>();

        // Generate 100 transaction of max

        int numTransaction = 50;
        int numItems = 5;
        for (int i = 0; i < numTransaction; i++)
        {
            SortedSet<int> t;
            do
            {
                t = new SortedSet<int>(Enumerable.Range(2, r.Next(numItems)).Select(x => r.Next(numItems)).ToList().Distinct().ToList());
            } while (t.Count < 2);
            dataset.Add(t);
        }

        // Create a new A-priori learning algorithm with the requirements
        var apriori = new Apriori(threshold: 3, confidence: 0);

        // Use the algorithm to learn a set matcher
        AssociationRuleMatcher<int> classifier = apriori.Learn(dataset.ToArray<SortedSet<int>>());

        // Use the classifier to find orders that are similar to 
        // orders where clients have bought items 1 and 2 together:
        int[][] matches = classifier.Decide(new[] { 1, 2 });

        foreach(var l in dataset)
        {
            Console.WriteLine(string.Join(",",l));
        }

        AssociationRule<int>[] rules = classifier.Rules;
        Console.ReadLine();
        //3 0 4
    }
}