
# @dataclass
# class CoCoCausalLMOutput(CausalLMOutput):
#     contexts: Optional[torch.FloatTensor]= None


    # @classmethod
    # def from_pretrained(
    #     cls,
    #     config_path_or_name: str,
    #     lora_path: str = None,
    #     cache_dir: str = None,
    # ):
    #     # 1-2줄로 끝날 건 아닌듯.
    #     # from_pretrained안에서 peft까지 전부 포함
    #     # 만약 from_pretraiend를 사용안하고 model을 load 한다면
    #     # CocomModel(config)이런식으로 접근하기.
    #     # from_pretrained는 순수히 사전에 저장된 cocom을 불러올때 사용하는 것?
    #     # 그럼 기본적으로 Model(config)을 진행한 후에 model.train()을 진행해야됨
    #     # 그럼 뭐가 문제냐 ? 문제 없음
    #     # peft_path가 존재한다고 했을때 from_pretrained를 이용해서 사전 학습된 것을 불러오는 것!
    #     # 오키도키 조쿠
    #     # config = Config.from_pretrained(config_path_or_name)
    #     # config.cache_dir = cache_dir
    #     # return cls(config=config)
    #     pass