using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

[System.Serializable]
public class save_data //class for save data
{
    public weight[] weights;
    //the index of the node is the same in nodes and in weights.
    //the second number represents the weight in regards to the node number in the next layer
    //eg weights[0][1] represents the weight of first input to the second node in first hidden layer.
    
    public bias[] biases;
    //index of bias is same as index of node in 'nodes'
}

[System.Serializable]
public class weight //dictionaries dont work, so doijg this
{
    [SerializeField]
    public float[] weight_array;
}

[System.Serializable]
public class bias //cant save dictionaries  :(
{
    [SerializeField]
    public float node_bias;
}

public class tim : MonoBehaviour
{
    private float e = 2.718281828459045f;
    public int hidden_layer_number; 
    public int node_number; //number of nodes in each layer
    public int output_number;
    public int input_number;
    private float[,] nodes; //contains values of each node where [0] = before sigmoid(net) and [1] = after sigmoid(out) and [2] = relationship between out and total error.
    private Dictionary<int, float[]> weight_adjustments = new Dictionary<int, float[]>();
    //same indexing as weights dictionary. used to store backpropogation changes
    private Dictionary<int, float> bias_adjustments = new Dictionary<int, float>();
    //indexing is the same as in biases;

    private int total; //used for some functions 
    private float largest_number; //used to find largest output
    private int largest_output;

    private float effect_of_weight;
    public float learning_rate;
    public int gen_counter = 0;
    private float effect_of_out;

    public float[] outputs;

    private string file_location;
    private string data_in_file;
    public save_data data; //data currently in use

    // Start is called before the first frame update
    void Start()
    {
        //finds file location
        file_location = Application.persistentDataPath + "/weights_biases.json";
        Debug.Log(file_location);


        //loads weights and biases from memory, creatse new if not there
        data = new save_data();
        if (File.Exists(file_location))
        {
            data_in_file = File.ReadAllText(file_location);
            data = JsonUtility.FromJson<save_data>(data_in_file);
        }
        else
        {
            restart();
        }

        outputs = new float[output_number];
        nodes = new float[input_number + node_number * hidden_layer_number + output_number, 3]; //sets length of 'nodes'

        //initialises adjustments
        for (int layer = 0; layer < hidden_layer_number + 1; layer++)
        {
            for (int current_node = 0; current_node < get_number_in_layer(layer); current_node++) //creates entry for each node in the dictionaries
            {
                weight_adjustments.Add(get_node(layer, current_node), new float[get_number_in_layer(layer+1)]);
                bias_adjustments.Add(get_node(layer, current_node), 0);
            }
        }
        for (int i = input_number + node_number * hidden_layer_number; i < output_number + input_number + node_number * hidden_layer_number; i++) //biases adjustments for final layer
        {
            bias_adjustments.Add(i, 0); 
        }
    }

    //resets save data to random
    public void restart()
    {
        data.weights = new weight[input_number + node_number * hidden_layer_number];
        data.biases = new bias[output_number + input_number + node_number * hidden_layer_number];
        //sets weights and biases to random stuff
        for (int layer = 0; layer < hidden_layer_number + 1; layer++)
        {
            for (int current_node = 0; current_node < get_number_in_layer(layer); current_node++) //creates entry for each node in the dictionaries
            {
                data.weights[get_node(layer, current_node)] = new weight();
                data.weights[get_node(layer, current_node)].weight_array = new float[get_number_in_layer(layer+1)];
                for (int j = 0; j < get_number_in_layer(layer+1); j++)
                {
                    data.weights[get_node(layer, current_node)].weight_array[j] = UnityEngine.Random.value * 2 -1; //sets each to random number between 0-1
                }
                data.biases[get_node(layer, current_node)] = new bias();
                data.biases[get_node(layer, current_node)].node_bias = UnityEngine.Random.value * 2 -1; //sets each to random number between 0-1
            }
        }
        for (int i = input_number + node_number * hidden_layer_number; i < output_number + input_number + node_number * hidden_layer_number; i++) //biases adjustments for final layer
        {
            data.biases[i] = new bias();
            data.biases[i].node_bias = UnityEngine.Random.value * 2 -1; 
        }
        save();
    }

    public void set_learning_rate(float new_rate)
    {
        learning_rate = new_rate;
    }

    //saves progress
    public void save()
    {
        data_in_file = JsonUtility.ToJson(data);
        File.WriteAllText(file_location, data_in_file);
    }

    //finds values of all net and out variable of nodes.
    private void fill_nodes(float[] inputs)
    {
        for (int i = 0; i<input_number; i++) //copies inputs into the nodes array
        {
            nodes[i, 0] = inputs[i];
            nodes[i, 1] = smol(nodes[i,0]);
        }
        //calculates the rest of the nodes
        for (int layer = 1; layer < hidden_layer_number + 2; layer++) //for each leayer
        {
            for (int current_node = 0; current_node < get_number_in_layer(layer); current_node++) //for the node it is currently calculating
            {
                nodes[get_node(layer, current_node), 0] = 0;
                for (int reference_node = 0; reference_node < get_number_in_layer(layer-1); reference_node++) //for the node who is being added
                {
                    //to the node being calculated: adds referenced node value * the weight stored in "weights"
                    nodes[get_node(layer, current_node), 0] += nodes[get_node(layer-1, reference_node), 1] * data.weights[get_node(layer-1, reference_node)].weight_array[current_node];
                }
                nodes[get_node(layer, current_node), 0] += data.biases[get_node(layer, current_node)].node_bias; //adds the bias
                nodes[get_node(layer, current_node),1] = smol(nodes[get_node(layer, current_node), 0]);
            }
        }
    }
    //processes each node
    public int think(float[] inputs)
    {
        //fills the nodes
        fill_nodes(inputs);

        //finds largest output
        largest_output = 0;
        largest_number = nodes[get_node(hidden_layer_number+1, 0), 0]; //sets it the to value of first output
        for (int i = 0; i < output_number; i++)
        {
            if (largest_number < nodes[get_node(hidden_layer_number+1, i), 0])
            {
                largest_output = i;
                largest_number = nodes[get_node(hidden_layer_number+1, i), 0];
            }
            outputs[i] = nodes[get_node(hidden_layer_number+1, i), 1];
        }
        //returns the thing
        return largest_output;
    }

    //dpes the backpropogation part, notes down changes it would make.
    public void back_prop_note(float[] inputs, float[] ideal_output)
    {
        fill_nodes(inputs);
        // clears (etotal / out) variables
        for (int current_node = 0; current_node < input_number + node_number * hidden_layer_number + output_number; current_node ++)
        {
            nodes[current_node, 2] = 0;
        }
        //output nodes relationship with error. (d Etotal / d out)
        for (int current_node = 0; current_node < output_number; current_node++)
        {
            //out - target is the realationship.
            nodes[get_node(hidden_layer_number + 1, current_node), 2] = 2f / output_number * ( nodes[get_node(hidden_layer_number+ 1, current_node), 1] - ideal_output[current_node] );
        }
        //finds weight changes
        for (int layer = hidden_layer_number + 1; layer > 0; layer--) //layer goes in reverse from output -> input
        {
            for (int current_node = 0; current_node < get_number_in_layer(layer); current_node++) // current node in the layer
            {
                for (int node_previous_layer = 0; node_previous_layer < get_number_in_layer(layer-1); node_previous_layer ++) //node from which weight is being calculated
                {
                    //calculate total error / weight
                    effect_of_weight = nodes[get_node(layer, current_node), 2]; // total error / out
                    effect_of_weight *= nodes[get_node(layer, current_node), 1] * (1f - nodes[get_node(layer, current_node), 1]);// out / net
                    effect_of_weight *= nodes[get_node(layer-1, node_previous_layer), 1];// net / weight
                    //adding to node adjust
                    weight_adjustments[get_node(layer-1, node_previous_layer)][current_node]-= learning_rate * effect_of_weight;
                    // update (total error / out) of next thing
                    effect_of_out = data.weights[get_node(layer-1, node_previous_layer)].weight_array[current_node]; //net o1 / out h1
                    effect_of_out *= nodes[get_node(layer, current_node), 2];//Eo1 / outo1
                    effect_of_out *= nodes[get_node(layer, current_node), 1] * (1f - nodes[get_node(layer, current_node), 1]);//outo1 / neto1
                    nodes[get_node(layer-1, node_previous_layer), 2] += effect_of_out;
                }
                //find bias changes
                effect_of_weight = nodes[get_node(layer, current_node), 2]; // total error / out
                effect_of_weight *= nodes[get_node(layer, current_node), 1] * (1f - nodes[get_node(layer, current_node), 1]);// out / net
                bias_adjustments[get_node(layer, current_node)] -= learning_rate * effect_of_weight;

            }
        }
    }
    //applies that changes
    public void back_prop_apply()
    {
        gen_counter += 1;
        //applies changes to biases., then clears them
        for (int i = 0; i < input_number + node_number * hidden_layer_number + output_number; i++)
        {
            data.biases[i].node_bias += bias_adjustments[i];
            bias_adjustments[i] = 0;
        }
        //aplies weight changes, then clear them
        for (int layer = 0; layer < hidden_layer_number + 1; layer++)
        {
            for (int current_node = 0; current_node < get_number_in_layer(layer); current_node++)
            {
                for (int next_node = 0; next_node < get_number_in_layer(layer+1); next_node++)
                {
                    data.weights[get_node(layer, current_node)].weight_array[next_node] += weight_adjustments[get_node(layer, current_node)][next_node];
                    weight_adjustments[get_node(layer, current_node)][next_node] = 0;
                }
            }
        }
    }

    //condenses a number to below 1 using sigmoid.
    private float smol(float x) 
    {
        return 1f / (1f + (Mathf.Pow(e,-x)));
    }

    //function to return the index of a node in the array(bc its confusing even for me)
    // layer 0 is first in input, number 0 is first in layer
    private int get_node(int layer, int number)
    {
        total = 0;
        for (int i = 0; i < layer; i++)
        {
            total += get_number_in_layer(i);
        }
        return total + number;
    }

    //finds number of items in a specified layer (0 = input layer)
    private int get_number_in_layer(int layer)
    {
        if (layer == 0)
        {
            return input_number;
        }
        else if (layer < hidden_layer_number+1)
        {
            return node_number;
        }
        else
        {
            return output_number;
        }
    }
}