
class Autoencoder(nn.Module):
    def __init__(self): 
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Linear(11, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(20, 20), 
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(20, 50),
        ).to(device)
        
        self.decoder = nn.Sequential(
                nn.Linear(50, 20),  
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(20, 11),
        ).to(device)


        self.optimizer = torch.optim.Adam(
            [
                        {'params' : self.encoder.parameters(), 'lr' : 5e-4, 'weight_decay': 0.001},
                        {'params' : self.decoder.parameters(), 'lr' : 5e-4, 'weight_decay': 0.001}
            ]
        )

        self.criterion = nn.MSELoss()


    def forward(self, data):
        encoded = self.encoder(data)
        result_data = self.decoder(encoded)
        
        return result_data

    def sparsity_loss(self, latent_z):
        return torch.mean(torch.abs(latent_z))
        
        
    def train_autoencoder(self): 
        self.train()
        train_loss_list = []
        val_loss_list = []
        num_epochs = 0
        delta = 0.00001
        encoder_best_weight = None 
        patient_limit = 10
        patient_check = 0
        epochs = 100
        best_params = None
        init_val_loss = float('inf')
        for epoch in range(epochs):
            tr_loss = 0
            for tr_batch in tqdm(tr_dataloader, total = len(tr_dataloader)):
                tr_x = tr_batch
                tr_x = tr_x.to(device)
                self.optimizer.zero_grad()
                pred_feature= self.forward(tr_x)
                # loss = self.criterion(pred_feature, tr_x) + self.sparsity_weight * self.sparsity_loss(pred_feature)
                loss = self.criterion(pred_feature, tr_x)
                tr_loss += loss.item()
    
                loss.backward()
                clip_grad_norm_(self.parameters(), max_norm= 1.0)
            
                self.optimizer.step()
            train_loss_list.append(tr_loss / len(tr_dataloader))
            print(f'epoch: {epoch + 1}, 훈련 손실: {(tr_loss / len(tr_dataloader)) : 2f}')
            
            self.eval()
            val_loss = 0
            print_loss = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_x = val_batch
                    val_x = val_batch.to(device)
                    preds = self.forward(val_x)
                    
                    loss = self.criterion(preds, val_x)
                    val_loss += loss.item()
                
                print_loss = val_loss / len(val_dataloader)
                val_loss_list.append(print_loss)
                print(f'epoch: {epoch + 1}, 검증 손실: {np.round(print_loss,6)}')
                
                if print_loss < init_val_loss:
                    num_epochs = epoch + 1
                    patient_check = 0
                    init_val_loss = print_loss
                    encoder_best_weight = self.encoder.state_dict()
                    decoder_best_weight = self.decoder.state_dict()
                else:
                    patient_check += 1
        
            if patient_check == patient_limit:
                print(f"가중치 저장 시점 Epoch: {num_epochs}")
                print(f"검증 손실: {init_val_loss} ")
                print('****학습 종료****')
                break
       
       
        return train_loss_list, val_loss_list
auto_encoder = Autoencoder()
