style_model = TransformerNet()
state_dict = torch.load(args.model)
# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
print('test_wwwwwwwwwwwwwwwwww')
print(list(state_dict.keys()))
print(list(state_dict))
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
style_model.load_state_dict(state_dict)
style_model.to(device)