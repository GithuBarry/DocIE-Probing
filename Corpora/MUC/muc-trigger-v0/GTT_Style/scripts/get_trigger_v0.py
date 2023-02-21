#by Barry Wang
import json
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

root = "../../../muc/processed/"
file = "dev.json"
f = open(root + file)
l = f.readline()
count = {}
count_w_perp = {}
count_type = Counter()
count_type_total = Counter()
selected_trigger = {
    'kidnapping': ["kidnap", "kidnapping", "kidnapped", "abducted", "forced", "forces", "hostage", "hostages",
                   "capture", "capture", "arrested"],
    'attack': ["attack", "attacked", "attacks", "murder", "murdered", "kill", "killed", "forced", "force",
               "assassinated", "assassination", "assassinate", "death", "shot", "shooting", "violence", "raid",
               "massacre", "dead"],
    'bombing': ["bomb", "bombed", "explosion", "bombs", "exploded", "explode", "explodes", "explosive", "dynamite",
                "blew", "detonated", "detonates", "blown", "mined", "grenades", "grenade", "blowing-up"],
    'arson': ["fire", "burned", "burn"],
    'robbery': ["rob", "force", "robbed", "robbing", "forced", "force", "stole"],
    'forced work stoppage': ["strike", "stoppage"]
}
lst = []
while l:
    j = json.loads(l)
    if j["templates"]:
        doctext = j['doctext'].split()
        for ii, template in enumerate(j["templates"]):
            count_type_total.update([template["incident_type"]])
            c = True
            vocab = set()
            triggers = []
            # Special case for bombing / attack
            if template["incident_type"] == "bombing / attack" or template["incident_type"] == "attack / bombing":
                for trigger in selected_trigger["bombing"] :
                    if trigger in j['doctext']:
                        j["templates"][ii]["Trigger"] = trigger
                        j["templates"][ii]["ManuallyLabeledTrigger"] = False
                        c = False
                        template["incident_type"] = "bombing"
                if(c):
                    template["incident_type"]= "attack"
                    for trigger in selected_trigger["attack"]:
                        if trigger in j['doctext']:
                            j["templates"][ii]["Trigger"] = trigger
                            j["templates"][ii]["ManuallyLabeledTrigger"] = False
                            c = False
                            template["incident_type"] = "bombing"
            #Normal case
            else:
                for trigger in selected_trigger[template["incident_type"]] :
                    if trigger in j['doctext']:
                        j["templates"][ii]["Trigger"] = trigger
                        j["templates"][ii]["ManuallyLabeledTrigger"] = False
                        c = False
                        break
            if (c):
                j["templates"][ii]["Trigger"] = "UNKNOWN"
                j["templates"][ii]["ManuallyLabeledTrigger"] = True
                count_type.update([template["incident_type"]])
                for w in doctext:
                    if w not in stopwords.words('english'):
                        # w = ps.stem(w)
                        if template["incident_type"] not in count:
                            count[template["incident_type"]] = Counter()
                        if w not in vocab:
                            count[template["incident_type"]].update([w])
                        vocab.add(w)

        # sents = j['doctext'].split(".")
        # for s in sents:
        #     for template in j["templates"]:
        #         add = False
        #         for mention in template["PerpInd"]:
        #             if s.find(mention[0][0]) != -1:
        #                 add = True
        #                 break
        #         for mention in template["PerpOrg"]:
        #             if add:
        #                 break
        #             if s.find(mention[0][0]) != -1:
        #                 add = True
        #                 break
        #         if add:
        #             words = s.split()
        #             for w in words:
        #                 if w not in stopwords.words('english'):
        #                     if template["incident_type"] not in count_w_perp:
        #                         count_w_perp[template["incident_type"]] = Counter()
        #                     count_w_perp[template["incident_type"]].update([w])
    lst.append(j)
    l = f.readline()

with open(root + 'triggered-' + file, 'w') as fp:
    json.dump(lst, fp)

print(count_type)
print(count_type_total)

for k in count:
    print("------\n", k, count[k])

print("-------\n\n\n\n")

for k in count_w_perp:
    print("------\n", k, count_w_perp[k])
