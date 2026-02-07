class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, loss_emotion, loss_intensity):
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])

        loss = (
            precision1 * loss_emotion + self.log_vars[0] +
            precision2 * loss_intensity + self.log_vars[1]
        )
        return loss
