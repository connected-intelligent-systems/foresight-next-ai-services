import json


class WakeUp:
    # TODO make sure all floats are json serializable
    def __init__(self, confid_asleep, confid_recent_wu, confid_awake, wu_time):
        self.confid_asleep = confid_asleep
        self.confid_recent_wu = confid_recent_wu
        self.confid_awake = confid_awake
        self.wu_time = wu_time

    def __str__(self):
        return "confidences: {conf}; wu_time: {wut}".format(
            conf=(self.confid_asleep,
                  self.confid_recent_wu,
                  self.confid_awake),
            wut=self.wu_time)

    def to_json(self):
        dict_repr = dict(
            confid_asleep=float(self.confid_asleep),
            confid_recent_wu=float(self.confid_recent_wu),
            confid_awake=float(self.confid_awake),
            wu_time=str(self.wu_time)
        )
        return json.dumps(dict_repr)
