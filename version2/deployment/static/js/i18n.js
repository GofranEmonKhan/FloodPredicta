// Bangladesh Context-Oriented Bilingual Dictionaries (English & authentic বাংলা)
const translations = {
    en: {
        app_title: "FloodPredicta | AI Flood Early Warning System",
        brand_name: "FloodPredicta",
        brand_tagline: "Multi-Station Flood Risk Assessment & Early Warning System for Bangladesh",
        day_mode: "Day Mode",
        night_mode: "Night Mode",
        audio_off: "River Ambience: Off",
        audio_on: "River Ambience: Playing",
        lang_switch: "বাংলা",
        
        // Panel 1: Parameters
        panel_inputs_title: "Hydrological & Climate Station Parameters",
        panel_inputs_desc: "Select a weather station and month, or stress-test with historical scenario presets.",
        presets_label: "Historical Scenario Presets:",
        preset_1998: "🌊 1998 Mega-Flood Deluge",
        preset_sylhet: "🌧️ Sylhet Haor Flash Inundation",
        preset_winter: "☀️ Normal Dry Winter (January)",
        
        station_label: "BMD Weather Station:",
        station_select_placeholder: "Choose a station...",
        station_meta_alt: "Elevation:",
        station_meta_coords: "Coordinates:",
        
        month_label: "Month & Season:",
        rainfall_label: "Current Month Rainfall (mm):",
        rainfall_hint: "Normal monthly average for selected station:",
        
        lags_toggle_title: "Antecedent Soil Moisture & Preceding Rainfall (Lags):",
        lag1_label: "Preceding Month 1 Rain (Lag-1, mm):",
        lag2_label: "Preceding Month 2 Rain (Lag-2, mm):",
        lags_autofill_btn: "Auto-fill Station Baselines",
        
        weather_grid_title: "Atmospheric & Thermal Indicators:",
        max_temp_label: "Max Temp (°C):",
        min_temp_label: "Min Temp (°C):",
        humidity_label: "Relative Humidity (%):",
        wind_label: "Wind Speed (m/s):",
        cloud_label: "Cloud Coverage (0-8):",
        sun_label: "Bright Sunshine (hrs/day):",
        
        mode_label: "Decision Operational Threshold Mode:",
        mode_balanced: "⚖️ Balanced Mode (Threshold 0.50 - Max F1 Score)",
        mode_alert: "🚨 Life-Safety Alert Mode (Threshold 0.35 - Max Recall ≥ 98%)",
        
        btn_predict: "Analyze Flood Risk",
        btn_predicting: "Computing Hydrological Vectors...",
        
        // Panel 2: Assessment
        panel_results_title: "Live Flood Risk Radar & Impact Assessment",
        panel_results_desc: "Real-time risk scoring and hydrological catchment assessment",
        risk_probability: "Flood Disaster Probability",
        status_label: "Operational Risk Status:",
        
        drivers_title: "Key Hydrological Risk Factors:",
        advisory_title: "Disaster Preparedness & Actionable Advisory:",
        
        empty_state_title: "Ready for Assessment",
        empty_state_desc: "Configure meteorological parameters on the left and click 'Analyze Flood Risk' to evaluate delta flood probability.",
        
        // Emergency Section
        emergency_title: "Emergency Helplines & Disaster Management Network (Bangladesh)",
        emergency_subtitle: "Instant direct-dial emergency support for residents, local administration, and rescue volunteers.",
        call_now: "Call Hotline",
        visit_portal: "Visit Official Portal",
        
        hl_999_title: "999 - National Emergency Service",
        hl_999_desc: "24/7 Police, Ambulance, Fire Service, and Flood Water Rescue operations across Bangladesh.",
        
        hl_1090_title: "1090 - Disaster Early Warning IVR",
        hl_1090_desc: "Toll-free interactive voice response providing real-time storm, monsoon flood, and weather bulletins.",
        
        hl_333_title: "333 - National Citizen Services",
        hl_333_desc: "Government relief distribution, emergency dry food allocation, and district administration assistance.",
        
        hl_16263_title: "16263 - Shastho Batayon",
        hl_16263_desc: "24/7 National Health Hotline for medical consultation and waterborne disease prevention.",
        
        hl_ffwc_title: "BWDB - Flood Forecasting & Warning Centre",
        hl_ffwc_desc: "Official river water level gauges, danger level benchmarks, and 5-day delta inundation maps."
    },
    bn: {
        app_title: "ফ্লাডপ্রেডিক্টা | বন্যা পূর্বাভাস ও দুর্যোগ সতর্কবার্তা",
        brand_name: "ফ্লাডপ্রেডিক্টা",
        brand_tagline: "বাংলাদেশের নদী অববাহিকাভিত্তিক বন্যা পূর্বাভাস ও দুর্যোগ ঝুঁকি নিরূপণ ব্যবস্থা",
        day_mode: "দিনের মোড",
        night_mode: "রাতের মোড",
        audio_off: "নদীর আবহ সঙ্গীত: বন্ধ",
        audio_on: "নদীর আবহ সঙ্গীত: বাজছে",
        lang_switch: "English",
        
        // Panel 1: Parameters
        panel_inputs_title: "জলবায়ু ও আবহাওয়া স্টেশন পরিমাপক",
        panel_inputs_desc: "আবহাওয়া স্টেশন ও মাস নির্বাচন করুন অথবা ঐতিহাসিক বন্যার পূর্বনির্ধারিত পরিস্থিতি পরীক্ষা করুন।",
        presets_label: "ঐতিহাসিক পরিস্থিতির উদাহরণ:",
        preset_1998: "🌊 ১৯৯৮ ঐতিহাসিক মহাবন্যা",
        preset_sylhet: "🌧️ সিলেট-সুনামগঞ্জ হাওর প্লাবন",
        preset_winter: "☀️ শুষ্ক শীতকাল (জানুয়ারি)",
        
        station_label: "বাংলাদেশ আবহাওয়া অধিদপ্তর স্টেশন:",
        station_select_placeholder: "স্টেশন বেছে নিন...",
        station_meta_alt: "সমুদ্রপৃষ্ঠ থেকে উচ্চতা:",
        station_meta_coords: "ভৌগোলিক স্থানাঙ্ক:",
        
        month_label: "মাস ও ঋতু:",
        rainfall_label: "বর্তমান মাসের বৃষ্টিপাত (মিলিমিটার):",
        rainfall_hint: "এই স্টেশনের সাধারণ মাসিক স্বাভাবিক বৃষ্টিপাত:",
        
        lags_toggle_title: "পূর্ববর্তী ২ মাসের বৃষ্টিপাত ও মাটির আর্দ্রতা সূচক (Lags):",
        lag1_label: "পূর্ববর্তী ১ম মাসের বৃষ্টিপাত (Lag-1, মিমি):",
        lag2_label: "পূর্ববর্তী ২য় মাসের বৃষ্টিপাত (Lag-2, মিমি):",
        lags_autofill_btn: "স্বাভাবিক মান বসান",
        
        weather_grid_title: "বায়ুমণ্ডলীয় ও তাপমাত্রার সূচকসমূহ:",
        max_temp_label: "সর্বোচ্চ তাপমাত্রা (°সে):",
        min_temp_label: "সর্বনিম্ন তাপমাত্রা (°সে):",
        humidity_label: "বাতাসের আপেক্ষিক আর্দ্রতা (%):",
        wind_label: "বাতাসের গতিবেগ (মি/সে):",
        cloud_label: "মেঘের ঘনত্ব (০-৮ স্কেল):",
        sun_label: "উজ্জ্বল সূর্যালোক (ঘণ্টা/দিন):",
        
        mode_label: "সিদ্ধান্ত গ্রহণের সতর্কতা মোড:",
        mode_balanced: "⚖️ সুষম মোড (কাট-অফ ০.৫০ - সর্বোত্তম নির্ভুলতা ও ভারসাম্য)",
        mode_alert: "🚨 জীবন-রক্ষা জরুরি মোড (কাট-অফ ০.৩৫ - সর্বোচ্চ ৯৮%+ শনাক্তকরণ)",
        
        btn_predict: "বন্যার ঝুঁকি নিরূপণ করুন",
        btn_predicting: "জলবায়ু সূচক বিশ্লেষণ করা হচ্ছে...",
        
        // Panel 2: Assessment
        panel_results_title: "সরাসরি বন্যা ঝুঁকি পরিমাপক ও বিশ্লেষণ",
        panel_results_desc: "রিয়েল-টাইম পূর্বাভাস ও নদী অববাহিকার সামগ্রিক হাইড্রোলজিক্যাল মূল্যায়ন",
        risk_probability: "বন্যার সম্ভাব্য ঝুঁকি মাত্রা",
        status_label: "বর্তমান সতর্কবার্তা স্তর:",
        
        drivers_title: "ঝুঁকি বৃদ্ধিকারী মূল জলবায়ু নিয়ামকসমূহ:",
        advisory_title: "জরুরি দুর্যোগ ব্যবস্থাপনা ও প্রস্তুতি সংক্রান্ত দিকনির্দেশনা:",
        
        empty_state_title: "বিশ্লেষণের জন্য প্রস্তুত",
        empty_state_desc: "বাম পাশের প্যানেলে আবহাওয়ার তথ্য প্রদান করে 'বন্যার ঝুঁকি নিরূপণ করুন' বোতামে চাপ দিন।",
        
        // Emergency Section
        emergency_title: "জরুরি সেবা ও জাতীয় দুর্যোগ সহায়তা কেন্দ্র (বাংলাদেশ)",
        emergency_subtitle: "সাধারণ নাগরিক, কৃষক, স্থানীয় প্রশাসন ও স্বেচ্ছাসেবকদের জন্য সার্বক্ষণিক জরুরি যোগাযোগ।",
        call_now: "সরাসরি কল করুন",
        visit_portal: "অফিসিয়াল পোর্টাল দেখুন",
        
        hl_999_title: "৯৯৯ - জাতীয় জরুরি সেবা",
        hl_999_desc: "সার্বক্ষণিক পুলিশ, অ্যাম্বুলেন্স, ফায়ার সার্ভিস এবং উদ্ধার তৎপরতার জন্য জাতীয় জরুরি হটলাইন।",
        
        hl_1090_title: "১০৯০ - দুর্যোগের আগাম বার্তা",
        hl_1090_desc: "দুর্যোগ ব্যবস্থাপনা ও ত্রাণ মন্ত্রণালয়ের টোল-ফ্রি ভয়েস মেসেজ ও আবহাওয়া পূর্বাভাস সেবা।",
        
        hl_333_title: "৩৩৩ - জাতীয় কল সেন্টার ও নাগরিক সেবা",
        hl_333_desc: "সরকারি জরুরি ত্রাণ সহায়তা, খাদ্য সামগ্রী বণ্টন এবং জেলা প্রশাসনের সাথে তাৎক্ষণিক যোগাযোগ।",
        
        hl_16263_title: "১৬২৬৩ - স্বাস্থ্য বাতায়ন",
        hl_16263_desc: "বন্যাজনিত পানিবাহিত রোগ ও জরুরি চিকিৎসার জন্য স্বাস্থ্য মন্ত্রণালয়ের ২৪/৭ টেলিমেডিসিন হটলাইন।",
        
        hl_ffwc_title: "বন্যা পূর্বাভাস ও সতর্কীকরণ কেন্দ্র (পাউবো)",
        hl_ffwc_desc: "বাংলাদেশ পানি উন্নয়ন বোর্ডের প্রধান নদ-নদীর পানি পরিমাপক ও ৫ দিনের বিস্তারিত প্লাবন চিত্র।"
    }
};

const stationNamesBn = {
    "Barisal": "বরিশাল",
    "Bhola": "ভোলা",
    "Bogra": "বগুড়া",
    "Chandpur": "চাঁদপুর",
    "Chittagong (City-Ambagan)": "চট্টগ্রাম (আমবাগান)",
    "Chittagong (IAP-Patenga)": "চট্টগ্রাম (পতেঙ্গা বিমানবন্দর)",
    "Comilla": "কুমিল্লা",
    "Cox's Bazar": "কক্সবাজার",
    "Dhaka": "ঢাকা",
    "Dinajpur": "দিনাজপুর",
    "Faridpur": "ফরিদপুর",
    "Feni": "ফেনী",
    "Hatiya": "হাতিয়া",
    "Ishurdi": "ঈশ্বরদী",
    "Jessore": "যশোর",
    "Khepupara": "খেপুপাড়া (পটুয়াখালী)",
    "Khulna": "খুলনা",
    "Kutubdia": "কুতুবদিয়া",
    "Madaripur": "মাদারীপুর",
    "Maijdee Court": "মাইজদী কোর্ট (নোয়াখালী)",
    "Mongla": "মংলা",
    "Mymensingh": "ময়মনসিংহ",
    "Patuakhali": "পটুয়াখালী",
    "Rajshahi": "রাজশাহী",
    "Rangamati": "রাঙ্গামাটি",
    "Rangpur": "রংপুর",
    "Sandwip": "সন্দ্বীপ",
    "Satkhira": "সাতক্ষীরা",
    "Sitakunda": "সীতাকুণ্ড",
    "Srimangal": "শ্রীমঙ্গল (মৌলভীবাজার)",
    "Sylhet": "সিলেট",
    "Tangail": "টাঙ্গাইল",
    "Teknaf": "টেকনাফ"
};

const monthNames = {
    en: [
        "1 - January (Winter)", "2 - February (Late Winter)", "3 - March (Early Summer)", 
        "4 - April (Summer / Kalbaishakhi)", "5 - May (Late Summer / Pre-Monsoon)", 
        "6 - June (Monsoon Entry)", "7 - July (Peak Monsoon / Shrabon)", 
        "8 - August (Peak Monsoon / Bhadro)", "9 - September (Late Monsoon)", 
        "10 - October (Autumn / Post-Monsoon)", "11 - November (Late Autumn)", 
        "12 - December (Winter)"
    ],
    bn: [
        "১ - জানুয়ারি (শীতকাল)", "২ - ফেব্রুয়ারি (শীতকাল)", "৩ - মার্চ (বসন্ত ও গ্রীষ্মের সূচনা)", 
        "৪ - এপ্রিল (গ্রীষ্মকাল ও কালবৈশাখী)", "৫ - মে (গ্রীষ্মকাল / প্রাক-বর্ষা)", 
        "৬ - জুন (বর্ষার সূচনা / আষাঢ়)", "৭ - জুলাই (তীব্র বর্ষা / শ্রাবণ)", 
        "৮ - আগস্ট (তীব্র বর্ষা / ভাদ্র)", "৯ - সেপ্টেম্বর (বর্ষার বিদায়)", 
        "১০ - অক্টোবর (শরৎকাল)", "১১ - নভেম্বর (হেমন্তকাল)", 
        "১২ - ডিসেম্বর (শীতকাল)"
    ]
};
