export const DataProfile = () => {

    const profileData = [
        { title: "Nombres", text: "Javier Gerardo" },
        { title: "Apellido", text: "Milei" },
        { title: "Email", text: "Javo@gmail.com" },
        { title: "DNI", text: "30.337.908" },
        { title: "Provincia", text: "Buenos Aires" },
        { title: "Circuito", text: "33" },
        { title: "Mesa", text: "1234" },
      ];

    return (
        <section className="flex flex-col w-full rounded-lg px-4 py-2 gap-x-4 bg-white">
            {profileData?.map((fielText, index) => (
                <FieldText fieldText={fielText} isLast={index === profileData.length - 1} key={crypto.randomUUID} />
            ))}
        </section>
    )
}

export const FieldText = ({fieldText, isLast}: any) => {
    return (
        <article className={`flex w-full justify-between ${isLast ? 'border-b-0' : 'border-b border-gray-300'}`}>
            <div className="w-2/4 flex justify-start py-4">
                <span className="font-semibold">{fieldText.title}</span>
            </div>

            <div className="w-2/4 flex justify-end py-4">
                <span className="font-normal">{fieldText.text}</span>
            </div>
        </article>
    )
}