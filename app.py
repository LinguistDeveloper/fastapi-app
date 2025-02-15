{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "16f8da2d-d4f0-4a54-b3ad-d27fd91efd36",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pip install transformers datasets evaluate rouge_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5116c23b-d705-4ca0-8e1a-83656a3cedca",
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e343401d-afcf-4cf3-8f70-0f1ec462d9af",
   "metadata": {},
   "outputs": [],
   "source": [
    "billsum = load_dataset(\"billsum\", split=\"ca_test\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "67126bc9-3225-4575-9e5e-c5f898d1ea16",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Dataset({\n",
       "    features: ['text', 'summary', 'title'],\n",
       "    num_rows: 1237\n",
       "})"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "billsum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6abb0360-7df9-4389-99db-9e03bd74d7c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "billsum = billsum.train_test_split(test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fa28570d-9c35-48fb-af34-4055cb3ffa02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DatasetDict({\n",
       "    train: Dataset({\n",
       "        features: ['text', 'summary', 'title'],\n",
       "        num_rows: 989\n",
       "    })\n",
       "    test: Dataset({\n",
       "        features: ['text', 'summary', 'title'],\n",
       "        num_rows: 248\n",
       "    })\n",
       "})"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "billsum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b07fe84b-38fd-4f33-9277-7b7225aa9ffb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'text': 'The people of the State of California do enact as follows:\\n\\n\\nSECTION 1.\\nSection 30515 of the Penal Code is amended to read:\\n30515.\\n(a) Notwithstanding Section 30510, “assault weapon” also means any of the following:\\n(1) A semiautomatic, centerfire rifle that does not have a fixed magazine but has any one of the following:\\n(A) A pistol grip that protrudes conspicuously beneath the action of the weapon.\\n(B) A thumbhole stock.\\n(C) A folding or telescoping stock.\\n(D) A grenade launcher or flare launcher.\\n(E) A flash suppressor.\\n(F) A forward pistol grip.\\n(2) A semiautomatic, centerfire rifle that has a fixed magazine with the capacity to accept more than 10 rounds.\\n(3) A semiautomatic, centerfire rifle that has an overall length of less than 30 inches.\\n(4) A semiautomatic pistol that does not have a fixed magazine but has any one of the following:\\n(A) A threaded barrel, capable of accepting a flash suppressor, forward handgrip, or silencer.\\n(B) A second handgrip.\\n(C) A shroud that is attached to, or partially or completely encircles, the barrel that allows the bearer to fire the weapon without burning the bearer’s hand, except a slide that encloses the barrel.\\n(D) The capacity to accept a detachable magazine at some location outside of the pistol grip.\\n(5) A semiautomatic pistol with a fixed magazine that has the capacity to accept more than 10 rounds.\\n(6) A semiautomatic shotgun that has both of the following:\\n(A) A folding or telescoping stock.\\n(B) A pistol grip that protrudes conspicuously beneath the action of the weapon, thumbhole stock, or vertical handgrip.\\n(7) A semiautomatic shotgun that has the ability to accept a detachable magazine.\\n(8) Any shotgun with a revolving cylinder.\\n(b) For purposes of this section, “fixed magazine” means an ammunition feeding device contained in, or permanently attached to, a firearm in such a manner that the device cannot be removed without disassembly of the firearm action.\\n(c) The Legislature finds a significant public purpose in exempting from the definition of “assault weapon” pistols that are designed expressly for use in Olympic target shooting events. Therefore, those pistols that are sanctioned by the International Olympic Committee and by USA Shooting, the national governing body for international shooting competition in the United States, and that were used for Olympic target shooting purposes as of January 1, 2001, and that would otherwise fall within the definition of “assault weapon” pursuant to this section are exempt, as provided in subdivision (d).\\n(d) “Assault weapon” does not include either of the following:\\n(1) Any antique firearm.\\n(2) Any of the following pistols, because they are consistent with the significant public purpose expressed in subdivision (c):\\nMANUFACTURER\\nMODEL\\nCALIBER\\nBENELLI\\nMP90\\n.22LR\\nBENELLI\\nMP90\\n.32 S&W LONG\\nBENELLI\\nMP95\\n.22LR\\nBENELLI\\nMP95\\n.32 S&W LONG\\nHAMMERLI\\n280\\n.22LR\\nHAMMERLI\\n280\\n.32 S&W LONG\\nHAMMERLI\\nSP20\\n.22LR\\nHAMMERLI\\nSP20\\n.32 S&W LONG\\nPARDINI\\nGPO\\n.22 SHORT\\nPARDINI\\nGP-SCHUMANN\\n.22 SHORT\\nPARDINI\\nHP\\n.32 S&W LONG\\nPARDINI\\nMP\\n.32 S&W LONG\\nPARDINI\\nSP\\n.22LR\\nPARDINI\\nSPE\\n.22LR\\nWALTHER\\nGSP\\n.22LR\\nWALTHER\\nGSP\\n.32 S&W LONG\\nWALTHER\\nOSP\\n.22 SHORT\\nWALTHER\\nOSP-2000\\n.22 SHORT\\n(3) The Department of Justice shall create a program that is consistent with the purposes stated in subdivision (c) to exempt new models of competitive pistols that would otherwise fall within the definition of “assault weapon” pursuant to this section from being classified as an assault weapon. The exempt competitive pistols may be based on recommendations by USA Shooting consistent with the regulations contained in the USA Shooting Official Rules or may be based on the recommendation or rules of any other organization that the department deems relevant.\\nSEC. 2.\\nSection 30680 is added to the Penal Code, to read:\\n30680.\\nSection 30605 does not apply to the possession of an assault weapon by a person who has possessed the assault weapon prior to January 1, 2017, if all of the following are applicable:\\n(a) Prior to January 1, 2017, the person was eligible to register that assault weapon pursuant to subdivision (b) of Section 30900.\\n(b) The person lawfully possessed that assault weapon prior to January 1, 2017.\\n(c) The person registers the assault weapon by January 1, 2018, in accordance with subdivision (b) of Section 30900.\\nSEC. 3.\\nSection 30900 of the Penal Code is amended to read:\\n30900.\\n(a) (1) Any person who, prior to June 1, 1989, lawfully possessed an assault weapon, as defined in former Section 12276, as added by Section 3 of Chapter 19 of the Statutes of 1989, shall register the firearm by January 1, 1991, and any person who lawfully possessed an assault weapon prior to the date it was specified as an assault weapon pursuant to former Section 12276.5, as added by Section 3 of Chapter 19 of the Statutes of 1989 or as amended by Section 1 of Chapter 874 of the Statutes of 1990 or Section 3 of Chapter 954 of the Statutes of 1991, shall register the firearm within 90 days with the Department of Justice pursuant to those procedures that the department may establish.\\n(2) Except as provided in Section 30600, any person who lawfully possessed an assault weapon prior to the date it was defined as an assault weapon pursuant to former Section 12276.1, as it read in Section 7 of Chapter 129 of the Statutes of 1999, and which was not specified as an assault weapon under former Section 12276, as added by Section 3 of Chapter 19 of the Statutes of 1989 or as amended at any time before January 1, 2001, or former Section 12276.5, as added by Section 3 of Chapter 19 of the Statutes of 1989 or as amended at any time before January 1, 2001, shall register the firearm by January 1, 2001, with the department pursuant to those procedures that the department may establish.\\n(3) The registration shall contain a description of the firearm that identifies it uniquely, including all identification marks, the full name, address, date of birth, and thumbprint of the owner, and any other information that the department may deem appropriate.\\n(4) The department may charge a fee for registration of up to twenty dollars ($20) per person but not to exceed the reasonable processing costs of the department. After the department establishes fees sufficient to reimburse the department for processing costs, fees charged shall increase at a rate not to exceed the legislatively approved annual cost-of-living adjustment for the department’s budget or as otherwise increased through the Budget Act but not to exceed the reasonable costs of the department. The fees shall be deposited into the Dealers’ Record of Sale Special Account.\\n(b) (1) Any person who, from January 1, 2001, to December 31, 2016, inclusive, lawfully possessed an assault weapon that does not have a fixed magazine, as defined in Section 30515, including those weapons with an ammunition feeding device that can be readily removed from the firearm with the use of a tool, shall register the firearm before January 1, 2018, but not before the effective date of the regulations adopted pursuant to paragraph (5), with the department pursuant to those procedures that the department may establish by regulation pursuant to paragraph (5).\\n(2) Registrations shall be submitted electronically via the Internet utilizing a public-facing application made available by the department.\\n(3) The registration shall contain a description of the firearm that identifies it uniquely, including all identification marks, the date the firearm was acquired, the name and address of the individual from whom, or business from which, the firearm was acquired, as well as the registrant’s full name, address, telephone number, date of birth, sex, height, weight, eye color, hair color, and California driver’s license number or California identification card number.\\n(4) The department may charge a fee in an amount of up to fifteen dollars ($15) per person but not to exceed the reasonable processing costs of the department. The fee shall be paid by debit or credit card at the time that the electronic registration is submitted to the department. The fee shall be deposited in the Dealers’ Record of Sale Special Account to be used for purposes of this section.\\n(5) The department shall adopt regulations for the purpose of implementing this subdivision. These regulations are exempt from the Administrative Procedure Act (Chapter 3.5 (commencing with Section 11340) of Part 1 of Division 3 of Title 2 of the Government Code).\\nSEC. 4.\\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\\u2009B of the California Constitution.',\n",
       " 'summary': '(1)\\xa0Existing law generally prohibits the possession or transfer of assault weapons, except for the sale, purchase, importation, or possession of assault weapons by specified individuals, including law enforcement officers. Under existing law, “assault weapon” means, among other things, a semiautomatic centerfire rifle or a semiautomatic pistol that has the capacity to accept a detachable magazine and has any one of specified attributes, including, for rifles, a thumbhole stock, and for pistols, a second handgrip.\\nThis bill would revise this definition of “assault weapon” to mean a semiautomatic centerfire rifle, or a semiautomatic pistol that does not have a fixed magazine but has any one of those specified attributes. The bill would also define “fixed magazine” to mean an ammunition feeding device contained in, or permanently attached to, a firearm in such a manner that the device cannot be removed without disassembly of the firearm action.\\nBy expanding the definition of an existing crime, the bill would impose a state-mandated local program.\\n(2)\\xa0Existing law requires that any person who, within this state, possesses an assault weapon, except as otherwise provided, be punished as a felony or for a period not to exceed one year in a county jail.\\nThis bill would exempt from punishment under that provision a person who possessed an assault weapon prior to January 1, 2017, if specified requirements are met.\\n(3)\\xa0Existing law requires that, with specified exceptions, any person who, prior to January 1, 2001, lawfully possessed an assault weapon prior to the date it was defined as an assault weapon, and which was not specified as an assault weapon at the time of lawful possession, register the firearm with the Department of Justice. Existing law permits the Department of Justice to charge a fee for registration of up to $20 per person but not to exceed the actual processing costs of the department. Existing law, after the department establishes fees sufficient to reimburse the department for processing costs, requires fees charged to increase at a rate not to exceed the legislatively approved annual cost-of-living adjustment for the department’s budget or as otherwise increased through the Budget Act. Existing law requires those fees to be deposited into the Dealers’ Record of Sale Special Account. Existing law, the Administrative Procedure Act, establishes the requirements for the adoption, publication, review, and implementation of regulations by state agencies.\\nThis bill would require that any person who, from January 1, 2001, to December 31, 2016, inclusive, lawfully possessed an assault weapon that does not have a fixed magazine, as defined, and including those weapons with an ammunition feeding device that can be removed readily from the firearm with the use of a tool, register the firearm with the Department of Justice before January 1, 2018, but not before the effective date of specified regulations. The bill would permit the department to increase the $20 registration fee as long as it does not exceed the reasonable processing costs of the department. The bill would also require registrations to be submitted electronically via the Internet utilizing a public-facing application made available by the department. The bill would require the registration to contain specified information, including, but not limited to, a description of the firearm that identifies it uniquely and specified information about the registrant. The bill would permit the department to charge a fee of up to $15 per person for registration through the Internet, not to exceed the reasonable processing costs of the department to be paid and deposited, as specified, for purposes of the registration program. The bill would require the department to adopt regulations for the purpose of implementing those provisions and would exempt those regulations from the Administrative Procedure Act. The bill would also make technical and conforming changes.\\n(4)\\xa0The California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\\nThis bill would provide that no reimbursement is required by this act for a specified reason.',\n",
       " 'title': 'An act to amend Sections 30515 and 30900 of, and to add Section 30680 to, the Penal Code, relating to firearms.'}"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "billsum[\"train\"][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "30930cfd-9679-4913-8cbc-18f396f9589c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7301fc7a-3f20-41d4-b7a9-c78ede3cfa7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "checkpoint = \"google-t5/t5-small\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(checkpoint)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ceeb39c7-eb9b-4199-bc9b-bea621c19cbb",
   "metadata": {},
   "outputs": [],
   "source": [
    "prefix = \"summarize: \"\n",
    "\n",
    "\n",
    "def preprocess_function(examples):\n",
    "    inputs = [prefix + doc for doc in examples[\"text\"]]\n",
    "    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)\n",
    "\n",
    "    labels = tokenizer(text_target=examples[\"summary\"], max_length=128, truncation=True)\n",
    "\n",
    "    model_inputs[\"labels\"] = labels[\"input_ids\"]\n",
    "    return model_inputs\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f0681966-4a00-4471-add8-7aed6645fbca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a2cce39db9134c6d959a80af9e9f7983",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/989 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a686eca4829a4d79ab9f2ca7fa3d8d4c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/248 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "tokenized_billsum = billsum.map(preprocess_function, batched=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "052e32b3-8670-40b3-be6c-0f89bf25c740",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import DataCollatorForSeq2Seq\n",
    "\n",
    "data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a7d002d0-9d02-4918-ae92-98674020c292",
   "metadata": {},
   "outputs": [],
   "source": [
    "import evaluate\n",
    "\n",
    "rouge = evaluate.load(\"rouge\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "1741185d-1a18-4106-bb1a-65e073fd4302",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "\n",
    "def compute_metrics(eval_pred):\n",
    "    predictions, labels = eval_pred\n",
    "    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)\n",
    "    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)\n",
    "    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)\n",
    "\n",
    "    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)\n",
    "\n",
    "    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]\n",
    "    result[\"gen_len\"] = np.mean(prediction_lens)\n",
    "\n",
    "    return {k: round(v, 4) for k, v in result.items()}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1674910f-0449-474b-8eaa-2248bec3319f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer\n",
    "\n",
    "model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "a3e96ced-0451-4358-812c-91e653be770d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting bitsandbytesNote: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "  Downloading bitsandbytes-0.45.2-py3-none-win_amd64.whl.metadata (5.9 kB)\n",
      "Requirement already satisfied: accelerate in c:\\users\\usuario\\anaconda3\\lib\\site-packages (1.3.0)\n",
      "Requirement already satisfied: torch<3,>=2.0 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from bitsandbytes) (2.5.1)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from bitsandbytes) (1.26.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from accelerate) (23.2)\n",
      "Requirement already satisfied: psutil in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from accelerate) (5.9.0)\n",
      "Requirement already satisfied: pyyaml in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from accelerate) (6.0.1)\n",
      "Requirement already satisfied: huggingface-hub>=0.21.0 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from accelerate) (0.27.1)\n",
      "Requirement already satisfied: safetensors>=0.4.3 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from accelerate) (0.5.2)\n",
      "Requirement already satisfied: filelock in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.21.0->accelerate) (3.13.1)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.21.0->accelerate) (2024.3.1)\n",
      "Requirement already satisfied: requests in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.21.0->accelerate) (2.32.2)\n",
      "Requirement already satisfied: tqdm>=4.42.1 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.21.0->accelerate) (4.66.4)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.21.0->accelerate) (4.11.0)\n",
      "Requirement already satisfied: networkx in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from torch<3,>=2.0->bitsandbytes) (3.2.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from torch<3,>=2.0->bitsandbytes) (3.1.4)\n",
      "Requirement already satisfied: setuptools in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from torch<3,>=2.0->bitsandbytes) (75.8.0)\n",
      "Requirement already satisfied: sympy==1.13.1 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from torch<3,>=2.0->bitsandbytes) (1.13.1)\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from sympy==1.13.1->torch<3,>=2.0->bitsandbytes) (1.3.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from tqdm>=4.42.1->huggingface-hub>=0.21.0->accelerate) (0.4.6)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from jinja2->torch<3,>=2.0->bitsandbytes) (2.1.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.21.0->accelerate) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.21.0->accelerate) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.21.0->accelerate) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\usuario\\anaconda3\\lib\\site-packages (from requests->huggingface-hub>=0.21.0->accelerate) (2024.8.30)\n",
      "Downloading bitsandbytes-0.45.2-py3-none-win_amd64.whl (69.1 MB)\n",
      "   ---------------------------------------- 0.0/69.1 MB ? eta -:--:--\n",
      "    --------------------------------------- 1.6/69.1 MB 7.6 MB/s eta 0:00:09\n",
      "   --- ------------------------------------ 6.8/69.1 MB 16.8 MB/s eta 0:00:04\n",
      "   ------- -------------------------------- 12.6/69.1 MB 20.2 MB/s eta 0:00:03\n",
      "   ---------- ----------------------------- 17.8/69.1 MB 21.6 MB/s eta 0:00:03\n",
      "   ------------- -------------------------- 23.6/69.1 MB 22.3 MB/s eta 0:00:03\n",
      "   ---------------- ----------------------- 28.8/69.1 MB 23.2 MB/s eta 0:00:02\n",
      "   ------------------- -------------------- 33.8/69.1 MB 23.3 MB/s eta 0:00:02\n",
      "   ---------------------- ----------------- 39.1/69.1 MB 23.4 MB/s eta 0:00:02\n",
      "   ------------------------ --------------- 42.7/69.1 MB 22.8 MB/s eta 0:00:02\n",
      "   --------------------------- ------------ 48.0/69.1 MB 23.1 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 53.2/69.1 MB 23.2 MB/s eta 0:00:01\n",
      "   --------------------------------- ------ 58.5/69.1 MB 23.4 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 63.4/69.1 MB 23.5 MB/s eta 0:00:01\n",
      "   ---------------------------------------  68.9/69.1 MB 23.8 MB/s eta 0:00:01\n",
      "   ---------------------------------------  68.9/69.1 MB 23.8 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 69.1/69.1 MB 21.2 MB/s eta 0:00:00\n",
      "Installing collected packages: bitsandbytes\n",
      "Successfully installed bitsandbytes-0.45.2\n"
     ]
    }
   ],
   "source": [
    "#pip install bitsandbytes accelerate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "65a55b08-4b13-4f84-9361-db501347ddcc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='15' max='15' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [15/15 2:59:32, Epoch 0/1]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "TrainOutput(global_step=15, training_loss=3.501862080891927, metrics={'train_runtime': 11445.9848, 'train_samples_per_second': 0.086, 'train_steps_per_second': 0.001, 'total_flos': 259856258826240.0, 'train_loss': 3.501862080891927, 'epoch': 0.967741935483871})"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"training_args = Seq2SeqTrainingArguments(\n",
    "    output_dir=\"my_awesome_billsum_model\",\n",
    "    eval_strategy=\"no\",\n",
    "    learning_rate=2e-5,\n",
    "    per_device_train_batch_size=16,\n",
    "    per_device_eval_batch_size=16,\n",
    "    weight_decay=0.01,\n",
    "    save_total_limit=3,\n",
    "    num_train_epochs=1,\n",
    "    predict_with_generate=True,\n",
    "    fp16=True, #change to bf16=True for XPU\n",
    "    push_to_hub=False,\n",
    ")\"\"\"\n",
    "\n",
    "training_args = Seq2SeqTrainingArguments(\n",
    "    output_dir=\"my_awesome_billsum_model\",\n",
    "    eval_strategy=\"no\",\n",
    "    learning_rate=2e-5,\n",
    "    per_device_train_batch_size=16,  # Increase batch size\n",
    "    per_device_eval_batch_size=16,\n",
    "    weight_decay=0.01,\n",
    "    save_total_limit=3,\n",
    "    num_train_epochs=1,  # Reduce epochs\n",
    "    predict_with_generate=True,\n",
    "    fp16=True,  # Keep mixed precision\n",
    "    push_to_hub=False,\n",
    "#    optim=\"adamw_bnb_8bit\",  # Use 8-bit optimizer\n",
    "    logging_steps=100,  # Reduce logging overhead\n",
    "    dataloader_num_workers=4,  # Speed up data loading\n",
    "    save_strategy=\"epoch\",  # Reduce checkpointing overhead\n",
    "    gradient_accumulation_steps=4  # Effective larger batch size\n",
    ")\n",
    "\n",
    "\n",
    "\"\"\"training_args = TrainingArguments(\n",
    "    output_dir=\"./tmp_test\",  # Temporary output directory\n",
    "    max_steps=2,  # Run only 2 steps\n",
    "    per_device_train_batch_size=1,  # Smallest batch size\n",
    "    per_device_eval_batch_size=1,  # Smallest batch size\n",
    "    evaluation_strategy=\"no\",  # No evaluation for speed\n",
    "    save_strategy=\"no\",  # No checkpoint saving\n",
    "    logging_strategy=\"no\",  # No logging\n",
    ")\"\"\"\n",
    "\n",
    "trainer = Seq2SeqTrainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=tokenized_billsum[\"train\"],\n",
    "    eval_dataset=tokenized_billsum[\"test\"],\n",
    "    processing_class=tokenizer,\n",
    "    data_collator=data_collator,\n",
    "    compute_metrics=compute_metrics,\n",
    ")\n",
    "\n",
    "trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "48cfe545-8991-4f3c-b24b-8d3ae21d85cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "#pip install huggingface_hub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "b108d0dd-2fc5-47b1-b990-4402d0691b99",
   "metadata": {},
   "outputs": [],
   "source": [
    "from huggingface_hub import login\n",
    "login(token=\"\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d410ff52-1be5-45ee-ab32-7518ead8040a",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"from huggingface_hub import notebook_login\n",
    "\n",
    "notebook_login(token=\"\")\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "6c8eac35-d57e-4343-86f4-dba341224b9f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0e3d5fa0b7f047e4a2415f1bd3499c6b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Upload 2 LFS files:   0%|          | 0/2 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6ba210b970174eaaacaa0d7eb93deaab",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/242M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "40a5650a919c4531a96c3fd4c0c87f56",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "training_args.bin:   0%|          | 0.00/5.50k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "CommitInfo(commit_url='https://huggingface.co/rcook/my_awesome_billsum_model/commit/9212f45b45879a160197881099039af93dcb72b1', commit_message='End of training', commit_description='', oid='9212f45b45879a160197881099039af93dcb72b1', pr_url=None, repo_url=RepoUrl('https://huggingface.co/rcook/my_awesome_billsum_model', endpoint='https://huggingface.co', repo_type='model', repo_id='rcook/my_awesome_billsum_model'), pr_revision=None, pr_num=None)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.push_to_hub()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6b03d61-19fa-40f4-959c-f7d9900f0e9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "text = \"summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "25cea8e2-5ee9-4c48-a11c-149e0429a75c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1d4f8999ccbe4d1b8414760ebd4ca3cb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/1.57k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Usuario\\anaconda3\\Lib\\site-packages\\huggingface_hub\\file_download.py:140: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\\Users\\Usuario\\.cache\\huggingface\\hub\\models--rcook--my_awesome_billsum_model. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.\n",
      "To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development\n",
      "  warnings.warn(message)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "64b6d385264c4807a4c6af90a51b3d24",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/242M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e79aeb8d95ba477ab86408e4503039d1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "generation_config.json:   0%|          | 0.00/149 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a3669a8029f545368f34fef4b7eefa1f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/21.7k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "38a791c019004d39a62e50f7b676fdf3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer.json:   0%|          | 0.00/2.42M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7d61413bd5a34ad89a1fd6eba14ca896",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "special_tokens_map.json:   0%|          | 0.00/2.67k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Device set to use cpu\n",
      "Your max_length is set to 200, but your input_length is only 103. Since this is a summarization task, where outputs shorter than the input are typically wanted, you might consider decreasing max_length manually, e.g. summarizer('...', max_length=51)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[{'summary_text': \"the Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs . it's the most aggressive action on tackling the climate crisis in american history . no one making under $400,000 per year will pay a penny more in taxes .\"}]"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from transformers import pipeline\n",
    "\n",
    "summarizer = pipeline(\"summarization\", model=\"rcook/my_awesome_billsum_model\")\n",
    "summarizer(text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "feb69526-8a2f-42b2-8d3a-d21a3291f134",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
