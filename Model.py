{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Model.py",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyMTTmz9OmfByNXOhVKcTFCV",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/susanzhang233/mollykill/blob/main/Model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eZBVPeV6pur-"
      },
      "source": [
        "# Setup\n",
        "To run DeepChem in Colab, we'll first need to run the following lines of code, these will download conda in the colab environment."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7LMTc_8uqK6d"
      },
      "source": [
        "# Graphic GAN Model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "j4ZsW2TnqHkE"
      },
      "source": [
        "## Preparations\n",
        "\n",
        "Now we are ready to import some useful packages."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GDVuZH_7rUxP"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FsmKzTT3jqM8"
      },
      "source": [
        "import deepchem as dc "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fe0qAgyBsIkh"
      },
      "source": [
        "Here we'll import our dataset for training. \n",
        "[base_ set](https://pubchem.ncbi.nlm.nih.gov/bioassay/1706)\n",
        "\n",
        "[dataset](https://github.com/yangkevin2/coronavirus_data)\n",
        "\n",
        "This dataset contains  In-vitro assay that detects inhibition of SARS-CoV 3CL protease via fluorescence from PubChem AID1706"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MQxqEQ0ssJ42"
      },
      "source": [
        "df = pd.read_csv('https://raw.githubusercontent.com/susanzhang233/mollykill/main/AID1706_binarized_sars.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        },
        "id": "cuCLFzO5tGff",
        "outputId": "a08a37b6-1e1d-4eea-82cf-2bb2caa2d491"
      },
      "source": [
        "df.head()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>smiles</th>\n",
              "      <th>activity</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>CC1=CC=C(O1)C(C(=O)NCC2=CC=CO2)N(C3=CC=C(C=C3)...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)S(=O)(=O)C3=C...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>CC1=CC2=C(C=C1)NC(=O)C(=C2)CN(CCC3=CC=CC=C3)CC...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>CC1=CC=C(C=C1)CN(C(C2=CC=CS2)C(=O)NCC3=CC=CO3)...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>CCN1C2=NC(=O)N(C(=O)C2=NC(=N1)C3=CC=CC=C3)C</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                              smiles  activity\n",
              "0  CC1=CC=C(O1)C(C(=O)NCC2=CC=CO2)N(C3=CC=C(C=C3)...         1\n",
              "1  CC1=CC=C(C=C1)S(=O)(=O)N2CCN(CC2)S(=O)(=O)C3=C...         1\n",
              "2  CC1=CC2=C(C=C1)NC(=O)C(=C2)CN(CCC3=CC=CC=C3)CC...         1\n",
              "3  CC1=CC=C(C=C1)CN(C(C2=CC=CS2)C(=O)NCC3=CC=CO3)...         1\n",
              "4        CCN1C2=NC(=O)N(C(=O)C2=NC(=N1)C3=CC=CC=C3)C         1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 137
        },
        "id": "bmftBd0UtO06",
        "outputId": "fe174141-3a1b-42ec-83d6-add600131c58"
      },
      "source": [
        "df.groupby('activity').count()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>smiles</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>activity</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>290321</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>405</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "          smiles\n",
              "activity        \n",
              "0         290321\n",
              "1            405"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wDLKQeFhtLXo"
      },
      "source": [
        "Observe the data above, it contains a 'smiles' column, which stands for the smiles representation of the molecules. There is also an 'activity' column, in which it is the label specifying whether that molecule is considered as hit for the protein."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tPFADN3bYRhK"
      },
      "source": [
        "## Train-test split"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1gmNrBO6N1Jr"
      },
      "source": [
        "### Select subpart for testing"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9XbkxsVegG8R"
      },
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nj19L4SmN52k"
      },
      "source": [
        "from sklearn.utils import resample\n",
        "true = df[df['activity']==1]\n",
        "false = df[df['activity']==0]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vFSW5tYXN55z"
      },
      "source": [
        "false_short = resample(false, n_samples=2000, replace = False)\n",
        "ihbt = pd.concat([false_short , true], ignore_index =  True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g7qU8glVYaDn"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ut3RUcwPvCr7"
      },
      "source": [
        "## Encoder"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "j_kac_Xz1Uq-"
      },
      "source": [
        "def featurizer(mol, max_length = 10):\n",
        "  '''\n",
        "  Parameters\n",
        "  -------\n",
        "  mol: rdkit molecule object\n",
        "  max_length: max_length of molecule to accept\n",
        "\n",
        "  Returns\n",
        "  ------\n",
        "  nodes: an array with atoms encoded by their atomic numbers\n",
        "  edges: a matrix indicating the bondtype between each of the included atoms\n",
        "  '''\n",
        "  nodes = []\n",
        "  edges = []\n",
        "  \n",
        " #initiate the encoder of bonds\n",
        "  bond_types = [\n",
        "          Chem.rdchem.BondType.ZERO,\n",
        "          Chem.rdchem.BondType.SINGLE,\n",
        "          Chem.rdchem.BondType.DOUBLE,\n",
        "          Chem.rdchem.BondType.TRIPLE,\n",
        "          Chem.rdchem.BondType.AROMATIC,\n",
        "        ]\n",
        "  encoder = {j:i for i, j in enumerate(bond_types,1)}\n",
        "  \n",
        "  #loop over the atoms within max_length\n",
        "  for i in range(max_length+1):\n",
        "    #append each atom's corresponding atomic number to the nodes array\n",
        "    nodes.append(mol.GetAtomWithIdx(i).GetAtomicNum())\n",
        "    \n",
        "    l = []\n",
        "    #loop over the atoms to generate a matrix\n",
        "    for j in range(max_length+1):\n",
        "      #get each of the bonds\n",
        "      current_bond = mol.GetBondBetweenAtoms(i,j)\n",
        "    \n",
        "      if current_bond == None:#some atoms are not connected\n",
        "        l.append(0)\n",
        "      else:\n",
        "        #if connected, encode that bond\n",
        "        l.append(encoder.get(current_bond.GetBondType()))\n",
        "\n",
        "    edges.append(l)#append each list to create a bond interaction matrix\n",
        "     \n",
        "  return nodes, edges\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KDPa7jbhyyY_"
      },
      "source": [
        "#def check_size(min):\n",
        "  #'''this function checks the length of the molecules and eliminate those that are too short'''\n",
        "input_df = ihbt['smiles']\n",
        "df_length = []\n",
        "for _ in input_df:\n",
        "  df_length.append(Chem.MolFromSmiles(_).GetNumAtoms() )\n",
        "#input_df = input_df.apply(Chem.MolFromSmiles)\n",
        "#ihbt['length'] = input_df.apply(GetNumAtoms)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DaihIG7rgkAH"
      },
      "source": [
        "ihbt['length'] = df_length"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LDmGtQoS1mBA"
      },
      "source": [
        "ihbt = ihbt[ihbt['length']>10]\n",
        "input_df = ihbt['smiles']\n",
        "input_df = input_df.apply(Chem.MolFromSmiles)\n",
        "input_df = input_df.apply(featurizer)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7HRi8p03gqVf"
      },
      "source": [
        "#X_train, X_test, y_train, y_test = train_test_split(input_df, ihbt['activity'], test_size = 0.3)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Vrmq5D2cKNeH"
      },
      "source": [
        "def de_featurizer(nodes, edges):\n",
        "  '''draw out a molecule\n",
        "  '''\n",
        "  mol = Chem.RWMol()\n",
        "  "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NxB150lou0Ew"
      },
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from tensorflow.keras import layers"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iaEnj6okg1e7"
      },
      "source": [
        "### Real Train Test Split"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mx3R0NyB-KQS"
      },
      "source": [
        "Since we'll be using multiple inputs, we'll create a tensorflow dataset to handle that for us."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Dt1p2tjdvzYC"
      },
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(input_df, ihbt['activity'], test_size = 0.2 )"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_syDmen8M2ZH"
      },
      "source": [
        "nodes_train, edges_train = list(zip(*x_train) )\n",
        "nodes_test, edges_test = list(zip(*x_test))\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IFX5RdAIfgVk"
      },
      "source": [
        "nd_train = np.array(nodes_train)\n",
        "eg_train = np.array(edges_train)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ixFQL52q1JF9"
      },
      "source": [
        "nd_test = np.array(nodes_test)\n",
        "eg_test = np.array(edges_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uIHdCJBCwQI8"
      },
      "source": [
        "-----------------------------"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "19jMVcr_BoNQ"
      },
      "source": [
        "### Train test splits finally!"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "K4XyjqSywTL8"
      },
      "source": [
        "---------------------"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ogXFAnm-uS6J"
      },
      "source": [
        "## Discriminator\n",
        "\n",
        "Now we are finally ready to build a discriminator of our GAN Model. Discriminator has the same logics as a common classification model."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yLkwvYLciDUa"
      },
      "source": [
        "def make_discriminator(num_atoms):\n",
        "  '''\n",
        "  create a discriminator model that takes in two inputs: nodes and edges of a single molecule\n",
        "  graphic neural network\n",
        "  '''\n",
        "  # This is the one!\n",
        "\n",
        "  #conv_node = layers.Conv2D(\n",
        "      #32, (3, 3), activation='relu', input_shape=(10, None)\n",
        "  #)\n",
        "  conv_edge = layers.Conv1D(32, (3,), activation = 'relu', input_shape = (num_atoms,num_atoms))\n",
        "  edges_tensor = keras.Input(shape = (num_atoms,num_atoms), name = 'edges')\n",
        "  x_edge = conv_edge(edges_tensor)\n",
        "  #x_edge = layers.MaxPooling1D((2,))(x_edge)\n",
        "  x_edge = layers.Conv1D(64, (3,), activation='relu')(x_edge)\n",
        "  x_edge = layers.Flatten()(x_edge)\n",
        "  x_edge = layers.Dense(64, activation = 'relu')(x_edge)\n",
        "\n",
        "  nodes_tensor = keras.Input(shape = (num_atoms,), name = 'nodes' )\n",
        "  x_node = layers.Dense(32, activation = 'relu' )(nodes_tensor)\n",
        "  x_node = layers.Dropout(0.2)(x_node)\n",
        "  x_node = layers.Dense(64, activation = 'relu')(nodes_tensor)\n",
        "\n",
        "  main = layers.concatenate([x_node,x_edge], axis = 1)\n",
        "  main = layers.Dense(32, activation='relu')(main)\n",
        "  output = layers.Dense(1, activation = 'sigmoid', name = 'label')(main)# number of classes\n",
        "\n",
        "  return keras.Model(\n",
        "    inputs = [nodes_tensor, edges_tensor],\n",
        "    outputs = output\n",
        "    )"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OqOY0_qLkbkt"
      },
      "source": [
        "discriminator = make_discriminator(11)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EwIhJ0F8uVQt"
      },
      "source": [
        "### Set the weight for training"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "a8A2gEO5uAft"
      },
      "source": [
        "hit_count, nonhit_count = np.bincount(ihbt['activity'])\n",
        "total_count = len(ihbt['activity'])\n",
        "weight_nonhit = (1 / nonhit_count) * (total_count) / 2.0\n",
        "weight_hit = (1 / hit_count) * (total_count) / 2.0\n",
        "class_weights = {0: weight_nonhit, 1: weight_hit}\n",
        "#Now, let’s use the weights when training our model:\n",
        "\n",
        "#model = build_model(X_train, metrics=METRICS)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Pa6ifm6sWwap",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "edac8daa-b197-443e-c30a-aa355b968956"
      },
      "source": [
        "discriminator.compile(optimizer='adam',\n",
        "              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),\n",
        "              metrics=['accuracy'])\n",
        "\n",
        "history = discriminator.fit([nd_train, eg_train],\n",
        "                    y_train,\n",
        "                    epochs=80, \n",
        "                    verbose = False,\n",
        "                    class_weight=class_weights\n",
        "                    #steps_per_epoch = 100,\n",
        "                    )"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/array_ops.py:5049: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/backend.py:5017: UserWarning: \"`binary_crossentropy` received `from_logits=True`, but the `output` argument was produced by a sigmoid or softmax activation and thus does not represent logits. Was this intended?\"\n",
            "  '\"`binary_crossentropy` received `from_logits=True`, but the `output`'\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "u9b4RwtRhSwF",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "outputId": "601bea08-cc31-40d4-a143-6af116b4825b"
      },
      "source": [
        " plt.plot(history.history[\"accuracy\"])\n",
        "plt.gca().set(xlabel = \"epoch\", ylabel = \"training accuracy\")"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[Text(0, 0.5, 'training accuracy'), Text(0.5, 0, 'epoch')]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 34
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "koanqJ4mf1nj",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ecf6a4e5-7ab5-4e73-d63f-b2445b01ad54"
      },
      "source": [
        "discriminator.evaluate([nd_test, eg_test], y_test, verbose = 2)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "16/16 - 0s - loss: 1.0464 - accuracy: 0.8046\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/backend.py:5017: UserWarning: \"`binary_crossentropy` received `from_logits=True`, but the `output` argument was produced by a sigmoid or softmax activation and thus does not represent logits. Was this intended?\"\n",
            "  '\"`binary_crossentropy` received `from_logits=True`, but the `output`'\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[1.046405553817749, 0.8045738339424133]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zbyOcyGd3Zga"
      },
      "source": [
        "def get_discriminator_loss(real_predictions,fake_predictions):\n",
        "    real_predictions = tf.sigmoid(real_predictions)#predictions of the real images\n",
        "    fake_predictions = tf.sigmoid(fake_predictions)#prediction of the images from the generator\n",
        "    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_predictions),real_predictions)# as such\n",
        "    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_predictions),fake_predictions)\n",
        "    return real_loss+ fake_loss"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CW-mxfFcBpHY",
        "outputId": "82ef54e7-89bf-4505-890a-73aaca11e1c5"
      },
      "source": [
        "nd_test.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(481, 11)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 38
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6tGqQfSkA5lT"
      },
      "source": [
        "## Generator"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JSaDd5eahjXc"
      },
      "source": [
        "BATCH_SIZE = len(nd_train)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yEDRIJsQPKWA"
      },
      "source": [
        "def make_generator(num_atoms, noise_input_shape):\n",
        "  '''create generator model\n",
        "  '''\n",
        "  inputs = keras.Input(shape = (noise_input_shape,))\n",
        "  x = layers.Dense(128, activation=\"tanh\")(inputs)# input_shape = (noise_input_shape,) )#256: filters\n",
        "  #x = layers.Dropout(0.2)(x)\n",
        "  x = layers.Dense(256,activation=\"tanh\")(x)\n",
        "  #x = layers.Dropout(0.2)(x)\n",
        "  x = layers.Dense(528,activation=\"tanh\")(x)\n",
        "\n",
        "  #generating edges\n",
        "  edges_gen = layers.Dense(units =num_atoms*num_atoms)(x)\n",
        "  edges_gen = layers.Reshape((num_atoms, num_atoms ))(edges_gen)\n",
        "\n",
        "  nodes_gen = layers.Dense(units = num_atoms)(x)\n",
        "  #assert nodes_gen.output_shape == (num_atoms)\n",
        "  #nodes_gen = layers.Reshape(num_atoms, num_atoms)(edges_gen)\n",
        "\n",
        "  #y = zeros(())\n",
        "  return keras.Model(\n",
        "    inputs = inputs,\n",
        "    outputs = [nodes_gen, edges_gen]\n",
        "    )\n",
        "\n",
        "  #return [nodes_gen, edges_gen]\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dBXhLm2QCXi0"
      },
      "source": [
        "generator = make_generator(11, 100)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JmablI9wyhoe",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 757
        },
        "outputId": "73e3ee43-575c-493e-e622-3c4615cfe002"
      },
      "source": [
        "keras.utils.plot_model(discriminator)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAV8AAALlCAIAAADsbnJYAAAABmJLR0QA/wD/AP+gvaeTAAAgAElEQVR4nOzdeUAU9f8/8PfsvQt7AHKoXIK3YHkjYlFqapapoOKRYVJ4lLfx+0jxNQP9GCp+PomVysdKS0H045WoqYV9TFBLRUHAIyVCRO5jV3Zh5/fH1EYwLIsCs7DPx1/O9Z7XzOw+fc/MMkPRNE0AABrgcV0AAJgppAMAsEM6AAA7pAMAsBOYON/mzZsvXLjQqqUAt4YPH758+XKuqwAzYmo6XLhwISUlxcfHp1WrAa6kpKRwXQKYHVPTgRDi4+Ozf//+1isFODR16lSuSwCzg+sOAMAO6QAA7JAOAMAO6QAA7JAOAMAO6QAA7JAOAMAO6QAA7JAOAMAO6QAA7JAOAMAO6QAA7JAOAMAO6QAA7DhIh5CQELlcTlHU1atX23K9KSkpffr04fF4FEU5OjpGRka22aoPHDjg4eFBURRFUU5OTrNnz26zVQM8sWY836Gl7Ny5c/To0TNmzGjj9fr4+Ny8eXPcuHEnT57MyspSqVRttuqAgICAgIDu3bsXFhbm5+e32XoBngbOLFqLRqPx9fXlugqAJ8dNOlAUZWQqTdP79+/fvn17m9XTGuLi4goKCriuAuDJtXA61NbWRkREuLq6SqXS/v37x8fHM+Npmo6Oju7Vq5dYLFYqlatWraq31Lp163r16iWVSjt16tStW7d169ZNmzbNeJvJyclDhw6VyWQKhcLb27u8vJwQcuLECYVCERUVZUq127Zts7Kykslkhw8fHj9+vEKhcHZ23rt3LzP13//+t0QicXBwmD9/fufOnSUSia+vb2pqKjN18eLFIpHIycmJGVy0aJGVlRVFUYWFhYSQpUuXrlix4s6dOxRFde/e3cS99+OPP/bt21epVEokEm9v75MnTxJCQkJCmAsWnp6eV65cIYTMnTtXJpMplcojR440tn8+/vhjmUwml8sLCgpWrFjRtWvXrKwsE8sA+ANtmsDAwMDAwCZnW7lypVgsTkxMLCkpWb16NY/Hu3TpEk3T4eHhFEVt2rSppKRErVbHxsYSQq5cucIsFRUVxefzDx8+rFarf/75Z0dHR39/f+NtVlZWKhSKDRs2aDSa/Pz8KVOmPHr0iKbpY8eOyeXytWvXNlbh2LFjCSElJSXMYHh4OCHkzJkzZWVlBQUFI0eOtLKy0mq1zNTQ0FArK6uMjIzHjx+np6cPGTJELpfn5OQwU2fNmuXo6GhoOTo6mhDClEHTdEBAgKenZ91Ve3p6KpVKI3tv//79a9asKS4uLioq8vHxsbOzMzTF5/N///13w5wzZ848cuRIk/ucELJkyZJPPvlkypQpN2/eNLJqE48vWJSWTAeNRiOTyYKCgphBtVotFosXLlyoVqtlMtmYMWMMczL/PxvSYciQIUOHDjVMffvtt3k8XnV1tZE2b9y4QQg5duyYifUbsKaDRqNhBpnYun37NjMYGhpa9/t86dIlQsiHH37IDLZ4OtS1bt06QkhBQQFN06dPnyaEREZGMpPKysp69OhRU1NDN75/Gm6acUgHaKglzyyysrLUarWXlxczKJVKnZycMjMzb9++rVarR40a1diCjx8/puu87Le2tlYoFPL5fCNtenh4ODg4zJ49e82aNffu3WupTRCJRIQQnU7HOnXw4MEymSwzM7OlVmeEUCgkhNTW1hJCXnzxxZ49e/7nP/9h9tK+ffuCgoKM7582qBA6vJZMh6qqKkLI+++/T/3p/v37arU6NzeXEGJvb9/Ygi+//PLPP/98+PBhjUZz+fLlQ4cOvfLKK8ynv7E2pVLp2bNn/fz8oqKiPDw8goKCNBpNC25LY8Ri8aNHj1qp8W+//dbf39/e3l4sFr/33nuG8RRFzZ8//+7du2fOnCGEfPXVV/PmzWMmNbZ/WqlCsCgtmQ7M9z8mJqZu5+TChQsSiYQQUl1d3diCa9asefHFF4ODgxUKxZQpU6ZNm7Zjxw7jbRJC+vXrd/To0by8vLCwsPj4+I0bN7bgtrDS6XSlpaXOzs4t2Oa5c+diYmIIITk5OZMnT3ZyckpNTS0rK9uwYUPd2YKDgyUSyc6dO7OyshQKhZubGzPeyP4BeEot+WsoFxcXiUTS8BeQXl5ePB4vOTl5wYIFrAump6ffuXPn0aNHAkH9ehprMy8vr7S0tG/fvvb29uvXrz916lRGRkZLbUhjfvjhB5qmDS8EEwgEjZ2DmO7nn3+2srIihFy/fl2n0y1cuNDDw4M0uOlrY2Mzffr0ffv2yeXyt956yzC+sf0D8PRasu8gkUjmzp27d+/ebdu2lZeX19bW5ubmPnjwwN7ePiAgIDExMS4urry8PC0trd5vGd555x1XV9fKykrT28zLy5s/f35mZqZWq71y5cr9+/eZL21SUpLpdzRNodfrS0pKampq0tLSli5d6urqGhwczEzq3r17cXHxoUOHdDrdo0eP7t+/X3dBW1vbvLy8e/fuVVRUsIaITqd7+PDhDz/8wKSDq6srIeT06dOPHz++deuW4dapwYIFC6qrq48dO/bqq682uX9aavPBopl49dLEa9rV1dVhYWGurq4CgYAJhfT0dJqmKyoqQkJC7OzsrK2t/fz8IiIiCCHOzs7Xrl2jafrs2bN2dnaGkoRCYZ8+fQ4cOGCkzXv37vn6+trY2PD5/C5duoSHhzPX8I8fPy6Xyw2X9+tKSUnp168fj8cjhDg5OUVFRcXGxspkMkJIjx497ty5s337doVCQQhxc3PLzs6maTo0NFQoFHbt2lUgECgUikmTJt25c8fQYFFR0QsvvCCRSLp16/buu+8yP+Lo3r07c8vzl19+cXNzk0qlfn5+n376qaenZ2OH4ODBg0yDYWFhtra2KpVq6tSpW7duJYR4enoabqDSND1gwIB//OMfpuzzDRs2SKVSQoiLi8vu3btb6viCRWnhdHgysbGxS5cuNQxWV1cvW7ZMLBar1epWWqOJQkNDbW1tua2hrpdffvnu3but0TLSARri4K+w6snPz1+8eHHdM2eRSOTq6qrT6XQ6HfMfIIeYe4oc0ul0zN3NtLQ0pp/CbT1gObj/KyypVCoUCuPi4h4+fKjT6fLy8nbu3BkREREUFMT08y1cWFjYrVu3srOz586d+9FHH3FdDlgQ7tNBqVSeOnXqxo0bPXv2lEqlffv23bVr1z//+c8vv/yS28JWr169a9eusrKybt26JSYmclWGTCbr3bv36NGj16xZ07dvX67KAAtE0XV+pGjE1KlTCSH79+9v5XqAGzi+0BD3fQcAME9IBwBgh3QAAHZIBwBgh3QAAHZIBwBgh3QAAHZIBwBgh3QAAHZIBwBgh3QAAHZIBwBgh3QAAHbNePpLSkoK85d80PGkpKQYnqYLwDA1HYYPH96qdZiby5cvE0IGDx7MdSFtxMfHx9IOMTTJ1Oc7WBrmHb8JCQlcFwLAGVx3AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2SAcAYId0AAB2FE3TXNdgFr744ostW7bU1tYyg48ePSKE2NvbM4N8Pn/p0qXBwcFclQfQ9pAOf8jKyurdu7eRGW7evGl8BoAOBmcWf+jVq5e3tzdFUQ0nURTl7e2NaABLg3T4y5w5c/h8fsPxAoHgjTfeaPt6ALiFM4u/5OXlOTs7N9whFEXl5OQ4OztzUhUAV9B3+EuXLl18fX15vL/tEx6P5+vri2gAC4R0+JvXX3+93qUHiqLmzJnDVT0AHMKZxd8UFxc7OjrW1NQYxvD5/IcPH9rZ2XFYFQAn0Hf4G1tb2zFjxggEAmaQz+ePGTMG0QCWCelQ3+zZs/V6PfNvmqZff/11busB4ArOLOqrqqrq1KnT48ePCSFisbiwsNDa2prrogA4gL5DfVZWVhMnThQKhQKBYNKkSYgGsFhIBxazZs2qqampra2dOXMm17UAcEZg+qy5ubk//fRT65ViPmprayUSCU3TlZWVCQkJXJfTFvCbDmioGdcdEhISpk+f3qrVAFfi4+OnTZvGdRVgXprRd2BYyFXM77//nqIof39/rgtpC6x/ewbQ7HSwEM8//zzXJQBwDOnArt5fWwBYIHwHAIAd0gEA2CEdAIAd0gEA2CEdAIAd0gEA2CEdAIAd0gEA2CEdAIAd0gEA2CEdAIAd0gEA2HWcdNDr9TExMb6+vkbmCQkJkcvlFEVdvXrV9JazsrLefffdfv36yeVygUCgVCp79uw5YcKECxcuPHXVJmHdtAMHDnh4eFB1iEQiBwcHf3//6OjokpKStqkNOrAOkg63bt167rnnli9frlarjcy2c+fOHTt2NKvluLg4b2/vtLS0zZs3//bbb1VVVVeuXPnoo49KS0uvX7/+dFWbpLFNCwgIuHv3rqenp1KppGlar9cXFBQkJCR069YtLCysX79+ly9fboPyoAPrCH/Bfe3atbVr1y5YsKCqqqplH06TkpISGhr6/PPPnzx50vCSCw8PDw8PD5VKdevWrRZcFyvTN42iKJVK5e/v7+/vP2HChOnTp0+YMCE7O1upVLZ2kdBRdYS+wzPPPHPgwIFZs2aJxeImZ27Wc5AiIyNra2vXr19viAaDsWPHvvPOO80rtPmatWkGgYGBwcHBBQUFn332WevVBh1eq6TD7t27Bw8eLJFIrKys3N3dP/roI0IITdObN2/u06ePWCy2sbGZNGlSZmYmM/+2bdusrKxkMtnhw4fHjx+vUCicnZ337t3LTO3Tpw9FUTweb9CgQUzv+r333lMqlRKJ5IsvvmiyGJqmo6Oje/XqJRaLlUrlqlWr6k49ceKEQqGIiopquKBWqz1z5oydnd3QoUObXAUnm2ZEcHAwISQpKelpGgFLR5ssPj7elPljYmIIIevXry8qKiouLv78889nzZpF03RERIRIJNq9e3dpaWlaWtrAgQM7deqUn5/PLBUeHk4IOXPmTFlZWUFBwciRI62srLRaLU3TNTU17u7urq6uNTU1hrUsW7YsJiam3qqHDRv2zDPP1BsZHh5OUdSmTZtKSkrUanVsbCwh5MqVK8zUY8eOyeXytWvXNtyQ7OxsQoiPj0+Tm8zVptE0bbjuUE95eTkhxMXFpcniaZomhMTHx5syJ1iUFk4HrVarUqleeOEFw5iampotW7ao1Wpra+ugoCDD+IsXLxJCDF9L5iuk0WiYQeY7fPv2bWaQSZyEhARmsKqqytXVtaysrN7aG36F1Gq1TCYbM2aMYQzz/7YhHYxgruqNHj3a+GxcbRqjsXSgaZq5EtHERtI0jXSARrTwmUVaWlppaenYsWMNY/h8/pIlS9LT0ysrKwcPHmwYP2TIEJFIlJqaytqOSCQihOh0OmYwJCREqVRu2bKFGdyzZ8+kSZMUCkWT9dy+fVutVo8aNeoJtoV5C5bxmyCEEK42zTjmKubTtwOWrIXTgenQqlSqeuNLS0vJn983A5VKVVFRYUqz1tbWb7/99k8//cT8t/zpp58uXrzYlAVzc3MJIfb29qbMXI+7u7tEImHOL4zgatOMY8ru3bv30zcFFquF06FLly6EkMLCwnrjmbyo94UpLS01/f1LixcvFgqFMTEx586dc3Fx8fT0NGUpiURCCKmurjZxLXWJxeKxY8cWFhaeP3++4dTi4uKQkBDC3aYZd+LECULI+PHjn74psFgtnA7u7u62tranTp2qN97Ly8va2rru73NSU1O1Wu2gQYNMbNnZ2XnatGmJiYkffPDB0qVLTVzKy8uLx+MlJyebOH89a9asEYvFy5cv12g09SbduHGDuc3J1aYZkZ+fHxMT4+zs/Oabbz59a2CxWjgdxGLx6tWrz507t3jx4t9//12v11dUVGRkZEgkkhUrVhw8eHDPnj3l5eXXr19fsGBB586dQ0NDTW98xYoVNTU1JSUlL774oomL2NvbBwQEJCYmxsXFlZeXp6Wlbd++ve4MSUlJjd3RJIQ8++yzX3/99Y0bN0aOHHn8+PGysjKdTvfrr7/u2LFj3rx5QqGQEMLVphnQNF1ZWanX62mafvToUXx8/IgRI/h8/qFDh3DdAZ6K6RcwTbyjSdP01q1bvb29JRKJRCIZMGBAbGwsTdN6vT46OrpHjx5CodDGxmby5MlZWVnM/LGxsTKZjBDSo0ePO3fubN++nflYu7m5ZWdn1235hRde2LlzZ73VXbhwYcSIEZ07d2a2yMnJydfXNzk5mZlaUVEREhJiZ2dnbW3t5+cXERFBCHF2dr527RpN08ePH5fL5ZGRkUY2JycnZ+XKld7e3tbW1nw+X6VSDRgwYN68eefPn2dm4GTTjhw50r9/f5lMJhKJmHfzMDcphg4dunbt2qKiIlOOFIPgngWwafZbdk2fH9oLiqLwll1oqCP8khoAWgPSAQDYIR0AgB3SAQDYIR0AgB3SAQDYIR0AgB3SAQDYIR0AgB3SAQDYIR0AgB3SAQDYIR0AgB3SAQDYIR0AgB3SAQDYIR0AgF2z37KbkJDQGnUAgLlpdjpMnz69NeoAAHPTjOdKWhTmIYvoKIElw3UHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGCHdAAAdkgHAGAn4LoAc5GcnJySkmIYzMzMJIRs2LDBMMbHx+f555/noDIAjlA0TXNdg1n47rvvXnrpJaFQyOPV70/p9XqdTnfq1KkxY8ZwUhsAJ5AOf6itrXV0dCwqKmKdamNjU1BQIBCgqwUWBNcd/sDn82fNmiUSiRpOEolEr7/+OqIBLA3S4S8zZszQarUNx2u12hkzZrR9PQDcwpnF37i5ueXk5NQb6ezsnJOTQ1EUJyUBcAV9h7+ZPXu2UCisO0YkEr3xxhuIBrBA6Dv8zc2bN/v27Vtv5PXr1728vDipB4BDSIf6+vbte/PmTcNg79696w4CWA6cWdQ3Z84cw8mFUCh84403uK0HgCvoO9SXk5Pj7u7O7BaKou7evevu7s51UQAcQN+hPldX18GDB/N4PIqihgwZgmgAi4V0YDFnzhwej8fn819//XWuawHgDM4sWDx69Khz586EkN9//93R0ZHrcgA4QtcRHx/PdTlgjuLj42mwPCx/O4CMIIQkJydTFPXcc89xXQj3pk+fznUJwA2WdJg2bVrb12Fuxo0bRwhRKBRcF8I9pIPFwt8dskMuAOCeBQCwQzoAADukAwCwQzoAADukAwCwQzoAADukAwCwQzoAADukAwCwQzoAADukAwCwQzoAADukAwCwa0/poNfrY2JifH19jcwTEhIil8spirp69WrLttyYrKysd999t1+/fnK5XCAQKJXKnj17Tpgw4cKFC0/Q2hNgLf7AgQMeHh5UHSKRyMHBwd/fPzo6uqSkpG1qg/at7qNgmOe+cPUgGuOys7NHjBhBCHnmmWeMz7l3715CyJUrV1q85YZ27twpFAqfe+65EydOlJSUPH78+M6dO/v27fP19f3888+b29oTMF68p6enUqmkaVqv15eUlHz//ffBwcEURXXu3PnSpUsmroLg2VCWqn083+HatWtr165dsGBBVVUV3aIPwnyallNSUkJDQ59//vmTJ08a3tDt4eHh4eGhUqlu3brVgnWyMr14iqJUKpW/v7+/v/+ECROmT58+YcKE7OxspVLZ2kVCO1Y3Ksy578AYNmxYk//D79u3jzSn72B6y/VMmDCBEJKamtqspVpDY8Ub+g71vPnmm4SQf/7zn6Y0TtB3sFRPeN1h9+7dgwcPlkgkVlZW7u7uH330ERMrmzdv7tOnj1gstrGxmTRpUmZmJjP/tm3brKysZDLZ4cOHx48fr1AonJ2dmVMAQkifPn0oiuLxeIMGDVKr1YSQ9957T6lUSiSSL774wpSAi46O7tWrl1gsViqVq1aterKNaujEiRMKhSIqKqrhJK1We+bMGTs7u6FDhzZZHie7xYjg4GBCSFJS0tM0Ah1f3agwse8QExNDCFm/fn1RUVFxcfHnn38+a9YsmqYjIiJEItHu3btLS0vT0tIGDhzYqVOn/Px8Zqnw8HBCyJkzZ8rKygoKCkaOHGllZaXVammarqmpcXd3d3V1rampMaxl2bJlMTEx9VbN+p9keHg4RVGbNm0q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++67559/nutaOJOZmTl8+PCJEyd++eWXXNdioZAO5oim6enTp589e/bSpUvdunXjuhwOFBUV+fj4ODo6njlzBpdguIJ0MFOVlZXDhw8XCATnz5+3tL8m0Ol0Y8eO/fXXX1NTUx0cHLgux3LhuoOZsra2Pnr0aG5u7ttvv811LW3tnXfeuXz58pEjRxAN3EI6mC93d/e9e/fGx8dv2rSJ61razsaNG3fu3Pn11197e3tzXYvFo8G8RUdH83i848ePc11IW0hKSuLz+Rs3buS6EKBpmsZ1h3bgzTffPHTo0MWLF7t37851La0oIyPD19d38uTJu3bt4roWIARXJduFx48fP/fccxUVFampqR31VVpFRUXDhg3r3Lnz6dOncZPCTOC6QzsgkUgOHDhQXFw8Z86cDpnmOp0uMDCwtrb24MGDiAbzgXRoH1xcXA4ePJiUlBQZGcl1LS1v0aJFP//889GjR/GALLOCdGg3RowYERMT83//93+JiYlc19KSNmzY8J///Oebb77x8vLiuhb4G1x3aGdCQ0P37t3bYV6llZSU9Oqrr27atGnJkiVc1wL1IR3aGZ1ON3r06AcPHly8eLG9v0orIyNj+PDhgYGBcXFxXNcCLJAO7c/Dhw8HDx7cp08f5tcBXJfzhAoLC4cNG9a1a9fTp0+LRCKuywEWuO7Q/jg6Oh4+fPh///tfeHg417U8ocePH0+cOJGm6YMHDyIazBbSoV0aOHDg559//vHHH+/du7fu+GvXrm3bto2rqlhptdqPP/64bheVpum33norPT396NGjnTp14rA2aAJnv9KEp7ZkyRKpVHr58mVmMD4+XiKRdO7cuba2ltvC6jpw4AAhJCAgoKqqihkTGRnJ5/O//fZbbguDJiEd2rGampqxY8e6ubnl5+f/4x//oCiKeZR7cnIy16X95eWXX+bz+QKBoH///rm5uQcPHuTxeFu3buW6Lmgarkq2b4WFhYMHD1Yqlenp6bW1tYQQoVAYHBy8fft2rksjhJCCgoIuXbowhQkEArlc/vjx47lz58bGxnJdGjQN1x3at5KSEj6ff/PmTeYbSAjR6XT79u2rrq7mtjDG7t27DW+mqampqaio0Gq1w4YN47YqMBHSoR1LSkoaOHBgTk6OTqerO76ysjIpKYmrquqKi4szxBYhpKampra29o033vh//+//6fV6DgsD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\n",
            "text/plain": [
              "<IPython.core.display.Image object>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9sWsSLGrIrwJ",
        "outputId": "c3cfd88f-d630-4830-fc49-c01f595db426"
      },
      "source": [
        "discriminator.predict(generator.predict(np.random.randint(0,2, size =(1,100))))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[7.631898e-05]], dtype=float32)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "m6LHhq5WS_Xh"
      },
      "source": [
        "def get_generator_loss(fake_predictions):\n",
        "  fake_predictions = tf.sigmoid(fake_predictions)#prediction of the images from the generator\n",
        "  fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_predictions),fake_predictions)\n",
        "  return fake_loss"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hQCy4VCXdRSs"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wlEzcxlhS_du"
      },
      "source": [
        "## GAN"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JATxa3CXflj1"
      },
      "source": [
        "#discriminator = make_discriminator(11)\n",
        "generator = make_generator(11, 100)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1edTtgMGcsYz"
      },
      "source": [
        "def train_step(mol,old_gen_loss,old_disc_loss):\n",
        "  fake_mol_noise = np.random.randn(batch_size, 100)# input for the generator\n",
        "  \n",
        "  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:\n",
        "\n",
        "    real_output = discriminator.evaluate(mol[:2])# trainable = False)\n",
        "\n",
        "    generated_mols = generator(fake_mol_noise)\n",
        "    fake_output = discriminator(generated_mols)\n",
        "\n",
        "    gen_loss = get_generator_loss(fake_output)\n",
        "    disc_loss = get_discriminator_loss(real_output, fake_output)\n",
        "\n",
        "        #optimizers would improve the performance of the model with these gradients\n",
        "    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)\n",
        "    gradients_of_disc= disc_tape.gradient(disc_loss, discriminator.trainable_variables)\n",
        "\n",
        "    generator_optimizer = tf.optimizers.Adam(1e-4)\n",
        "    discriminator_optimizer = tf.optimizers.Adam(1e-4)\n",
        "    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))\n",
        "    discriminator_optimizer.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))\n",
        "\n",
        "    print('generator loss: ', np.mean(gen_loss))\n",
        "    print('discriminator loss', np.mean(disc_loss))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6vVq46suS_gU"
      },
      "source": [
        "def train(nodes,edges, epochs):\n",
        "  for _ in range(epochs):\n",
        "    gen_loss = 0\n",
        "    disc_loss = 0\n",
        "    for n,e in zip(nodes, edges):\n",
        "      train = [n.reshape(1,11),e.reshape(1,11,11)]\n",
        "      train_step(train,gen_loss,disc_loss)\n",
        "    display.clear_output(wait=True)\n",
        "    #if (epoch+1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "si__14myfywQ"
      },
      "source": [
        "\n",
        "batch_size = 20\n",
        "train(nd_train,eg_train, 2)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zhCK9LzES_kz"
      },
      "source": [
        "#batch_size = 20\n",
        "#train(x_train , 30)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LMDuGLKvhT7Z"
      },
      "source": [
        "generator(np.random.randint(0,2, size =(1,100)))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-xjvOaNNnUIi"
      },
      "source": [
        "discriminator(generator(np.random.randint(0,2, size =(1,100))))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OY3wWyxfpPpS"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}