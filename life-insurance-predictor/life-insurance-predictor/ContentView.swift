import SwiftUI

struct ContentView: View {
    
    // MARK: - State variables for user inputs
    @State private var weight: String = ""
    @State private var height: String = ""
    @State private var sysBP: String = ""
    @State private var sexIndex: Int = 0
    @State private var smokerIndex: Int = 0
    @State private var nicOtherIndex: Int = 0
    @State private var numMeds: String = ""
    @State private var occupDangerIndex: Int = 0
    @State private var lsDangerIndex: Int = 0
    @State private var cannabisIndex: Int = 0
    @State private var opioidsIndex: Int = 0
    @State private var otherDrugsIndex: Int = 0
    @State private var drinksAWeek: String = ""
    @State private var addictionIndex: Int = 0
    @State private var majorSurgeryNum: String = ""
    @State private var diabetesIndex: Int = 0
    @State private var hdsIndex: Int = 0
    @State private var cholesterol: String = ""
    @State private var asthmaIndex: Int = 0
    @State private var immuneDefIndex: Int = 0
    @State private var familyCancerIndex: Int = 0
    @State private var familyHeartDiseaseIndex: Int = 0
    @State private var familyCholesterolIndex: Int = 0
    
    @State private var age: Double = 25
    @State private var policyAmount: Double = 125000
    @State private var paymentIndex: Int = 0
    
    // For showing result
    @State private var resultText: String = ""
    
    // MARK: - Arrays for pickers
    let sexOptions = ["m", "f"]
    let ynOptions = ["y", "n"]
    let riskOptions = ["1", "2", "3"]
    let paymentOptions = ["Lump", "Annual", "Monthly", "Compare Options"]
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 15) {
                    
                    Text("Death Predictors: Life Insurance Calculator")
                        .font(.title2)
                        .bold()
                    
                    Group {
                        Text("Please enter your personal info and risk factors:")
                            .font(.headline)
                        
                        // Weight
                        HStack {
                            Text("Weight (lbs):")
                            Spacer()
                            TextField("Weight", text: $weight)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                        
                        // Height
                        HStack {
                            Text("Height (in):")
                            Spacer()
                            TextField("Height", text: $height)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                        
                        // Sys BP
                        HStack {
                            Text("Sys_BP:")
                            Spacer()
                            TextField("Sys BP", text: $sysBP)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                        
                        // Number of medications
                        HStack {
                            Text("Number of Medications:")
                            Spacer()
                            TextField("Number", text: $numMeds)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                        
                        // Drinks per week
                        HStack {
                            Text("Drinks/week:")
                            Spacer()
                            TextField("Drinks/week", text: $drinksAWeek)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                        
                        // Major surgery
                        HStack {
                            Text("Number of major surgeries:")
                            Spacer()
                            TextField("Number", text: $majorSurgeryNum)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                        
                        // Cholesterol
                        HStack {
                            Text("Cholesterol:")
                            Spacer()
                            TextField("Cholesterol", text: $cholesterol)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                        
                        // Sex
                        Text("Sex:")
                        Picker("Sex", selection: $sexIndex) {
                            ForEach(0..<sexOptions.count) { i in
                                Text("\(sexFormat(sexOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Smoker
                        Text("Do you smoke?")
                        Picker("Do you smoke?", selection: $smokerIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Nic Other
                        Text("Do you use other forms of nicotine?")
                        Picker("Other forms of nicotine?", selection: $nicOtherIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Occup Danger
                        Text("How would you describe your occupational danger?")
                        Picker("Occupational Danger", selection: $occupDangerIndex) {
                            ForEach(0..<riskOptions.count) { i in
                                Text("\(risk_num_format(riskOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Lifestyle Danger
                        Text("How would you describe your lifestyle danger?")
                        Picker("Lifestyle Danger", selection: $lsDangerIndex) {
                            ForEach(0..<riskOptions.count) { i in
                                Text("\(risk_num_format(riskOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Cannabis
                        Text("Do you use cannabis, weed, or pot?")
                        Picker("Use cannabis?", selection: $cannabisIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Opioids
                        Text("Do you use opioids?")
                        Picker("Use opioids?", selection: $opioidsIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Other drugs
                        Text("Do you use any other drugs?")
                        Picker("Other drugs?", selection: $otherDrugsIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Addiction
                        Text("Have you ever had a history of addiction?")
                        Picker("History of addiction?", selection: $addictionIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Diabetes
                        Text("Do you have diabetes?")
                        Picker("Do you have diabetes?", selection: $diabetesIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Heart disease / stroke
                        Text("Do you have a history of heart disease or stroke?")
                        Picker("History of heart disease/stroke?", selection: $hdsIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Asthma
                        Text("Do you have asthma?")
                        Picker("Do you have asthma?", selection: $asthmaIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Immune deficiency
                        Text("Do you have an immune deficiency?")
                        Picker("Immune deficiency?", selection: $immuneDefIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Family history of cancer
                        Text("Do you have a family history of cancer?")
                        Picker("Family history of cancer?", selection: $familyCancerIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Family history of heart disease
                        Text("Do you have a family history of heart disease or stroke?")
                        Picker("Family history of heart disease/stroke?", selection: $familyHeartDiseaseIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        // Family cholesterol
                        Text("Do you have a family history of high cholesterol?")
                        Picker("Family history of high cholesterol?", selection: $familyCholesterolIndex) {
                            ForEach(0..<ynOptions.count) { i in
                                Text("\(ynFormat(ynOptions[i]))").tag(i)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                    }
                    
                    // Age
                    VStack {
                        Text("Current Age: \(Int(age))")
                        Slider(value: $age, in: 0...79, step: 1)
                    }
                    
                    // Policy Amount
                    VStack {
                        Text("Policy Amount: \(Int(policyAmount))")
                        Slider(value: $policyAmount, in: 0...500000, step: 5000)
                    }
                    
                    // Payment Type
                    Picker("Payment Type", selection: $paymentIndex) {
                        ForEach(0..<paymentOptions.count) { i in
                            Text(paymentOptions[i]).tag(i)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    
                    // MARK: - Calculate Button
                    Button(action: {
                        self.resultText = calculateInsurance()
                    }) {
                        Text("Calculate")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.vertical, 12)
                            .frame(maxWidth: .infinity)
                            .background(Color.blue)
                            .cornerRadius(8)
                    }
                    .padding(.vertical, 16)
                    
                    // MARK: - Display the Result
                    if !resultText.isEmpty {
                        ZStack {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.gray.opacity(0.15))  // Faded background
                                .shadow(radius: 2)
                            
                            VStack(spacing: 8) {
                                Text("Estimated Cost")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                
                                Text(resultText)
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .foregroundColor(.blue)
                            }
                            .padding()
                        }
                        .frame(maxWidth: .infinity, minHeight: 100)
                        .padding(.vertical, 8)
                    }
                    
                }
                .padding()
            }
            .navigationTitle("Life Insurance Calculator")
        }
    }
    
    // MARK: - Helper Functions
    
    func sexFormat(_ value: String) -> String {
        return value == "m" ? "Male" : "Female"
    }
    
    func ynFormat(_ value: String) -> String {
        return value == "y" ? "Yes" : "No"
    }
    
    func risk_num_format(_ value: String) -> String {
        switch value {
        case "1": return "Low (1)"
        case "2": return "Moderate (2)"
        case "3": return "High (3)"
        default:  return value
        }
    }
    
    func calculateInsurance() -> String {
        // Convert text fields to floats/doubles
        let w = Double(weight) ?? 0
        let h = Double(height) ?? 0
        let bp = Double(sysBP) ?? 0
        let meds = Double(numMeds) ?? 0
        let drWeek = Double(drinksAWeek) ?? 0
        let surgeryNum = Double(majorSurgeryNum) ?? 0
        let chol = Double(cholesterol) ?? 0
        
        // Example numeric encoding of pickers
        let sexVal = sexIndex == 0 ? 0.0 : 1.0
        let smokeVal = smokerIndex == 0 ? 1.0 : 0.0
        // etc. for other indexes...
        
        // Simple example logic
        let baseRisk = (w + h + bp + meds + drWeek + surgeryNum + chol) / 1000.0
        let ageFactor = age * 0.01
        let randomAdjustment = Double.random(in: -0.1...0.1)
        
        let finalRisk = baseRisk + ageFactor + randomAdjustment
        
        let payType = paymentOptions[paymentIndex]
        let policy = policyAmount
        
        let costEstimate = policy * finalRisk * 0.01  // dummy formula
        
        return String(format: "$%.2f", costEstimate)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
